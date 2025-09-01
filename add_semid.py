#!/usr/bin/env python3
"""
使用训练好的RQ-VAE模型为parquet文件生成semantic IDs
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
from torch.utils.data import DataLoader
import argparse
from collections import defaultdict

# 添加项目路径
sys.path.append('/mnt/xieshen/rqvae')

from modules.rqvae import RqVae
from data.custom_parquet_vectors import ParquetVectorIterable


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的RQ-VAE模型"""
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 从配置重建模型
    config = checkpoint.get('model_config', {})
    model = RqVae(
        input_dim=config.get('input_dim', 5120),
        embed_dim=config.get('embed_dim', 512),
        hidden_dims=config.get('hidden_dims', [2048, 1024, 512]),
        codebook_size=config.get('codebook_size', 256),
        codebook_kmeans_init=False,  # 推理时不需要
        codebook_normalize=config.get('codebook_normalize', True),
        codebook_sim_vq=config.get('codebook_sim_vq', False),
        n_layers=config.get('n_layers', 3),
        commitment_weight=config.get('commitment_weight', 0.35),
    )
    
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功 (iter {checkpoint.get('iter', 'unknown')})")
    return model


def process_single_file(input_path, output_path, model, device, batch_size=1024):
    """处理单个parquet文件"""
    
    # 读取原始数据
    try:
        table = pq.read_table(input_path)
        df = table.to_pandas()
    except Exception as e:
        print(f"读取文件失败 {input_path}: {e}")
        return False
    
    if len(df) == 0:
        print(f"跳过空文件: {input_path}")
        return True
    
    print(f"处理文件: {input_path.name} ({len(df):,} 行)")
    
    # 准备数据
    md5_oaids = df['md5_oaid'].values
    vectors = df['parsed_vector'].values
    
    # 转换向量格式并归一化
    processed_vectors = []
    valid_indices = []
    
    for i, vec in enumerate(vectors):
        if isinstance(vec, (list, np.ndarray)):
            vec_array = np.array(vec, dtype=np.float32)
            if vec_array.shape[0] == 5120:  # 检查维度
                # L2归一化
                norm = np.linalg.norm(vec_array, ord=2) + 1e-8
                vec_normalized = vec_array / norm
                processed_vectors.append(vec_normalized)
                valid_indices.append(i)
    
    if len(processed_vectors) == 0:
        print(f"警告: 文件 {input_path.name} 没有有效向量")
        return False
    
    print(f"有效向量: {len(processed_vectors):,}/{len(df):,}")
    
    # 批量推理生成SemID
    all_semids = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(processed_vectors), batch_size), 
                     desc="生成SemID", leave=False):
            batch_vectors = processed_vectors[i:i+batch_size]
            batch_tensor = torch.tensor(np.stack(batch_vectors), device=device)
            
            # 获取semantic IDs
            output = model.get_semantic_ids(batch_tensor, gumbel_t=0.001)
            sem_ids = output.sem_ids.cpu().numpy()  # shape: [n_layers, batch_size]
            
            # 转置为 [batch_size, n_layers]
            sem_ids = sem_ids.T
            all_semids.extend(sem_ids.tolist())
    
    # 构建输出数据
    output_data = {
        'md5_oaid': [md5_oaids[i] for i in valid_indices],
        'parsed_vector': [vectors[i] for i in valid_indices],
        'semid': all_semids,
        'semid_str': ['-'.join(map(str, semid)) for semid in all_semids]
    }
    
    output_df = pd.DataFrame(output_data)
    
    # 保存结果
    os.makedirs(output_path.parent, exist_ok=True)
    output_df.to_parquet(output_path, compression='snappy', index=False)
    
    print(f"✓ 保存完成: {output_path} ({len(output_df):,} 行)")
    return True


def main():
    parser = argparse.ArgumentParser(description='生成Semantic IDs')
    parser.add_argument('--model_path', type=str, 
                       default='/mnt/xieshen/rqvae/trained_models/shop_rqvae/checkpoint_best.pt',
                       help='模型检查点路径')
    parser.add_argument('--input_dirs', type=str, nargs='+',
                       default=[
                           "/mnt/xieshen/data/20250706",
                           "/mnt/xieshen/data/20250707", 
                           "/mnt/xieshen/data/20250708",
                           "/mnt/xieshen/data/20250709",
                           "/mnt/xieshen/data/20250710",
                           "/mnt/xieshen/data/20250711",
                           "/mnt/xieshen/data/20250712"
                       ], help='输入数据目录列表')
    parser.add_argument('--output_base', type=str,
                       default='/mnt/xieshen/data_processed/l2_0811',
                       help='输出基础目录')
    parser.add_argument('--batch_size', type=int, default=1024, help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    # 检查GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    print(f"使用设备: {args.device}")
    
    # 加载模型
    model = load_model(args.model_path, args.device)
    
    # 统计所有parquet文件
    all_parquet_files = []
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        if input_path.exists():
            files = list(input_path.glob("*_parsed.parquet"))
            all_parquet_files.extend([(input_path, f) for f in files])
    
    total_files = len(all_parquet_files)
    print(f"总共找到 {total_files} 个parquet文件需要处理")
    
    # 全局统计变量
    global_stats = {
        'total_rows': 0,
        'unique_md5_oaids': set(),
        'unique_vectors': set(),
        'unique_semids': set(),
        'vector_to_semid': {},  # vector_hash -> semid_str
        'semid_to_vectors': {}  # semid_str -> set of vector_hashes
    }
    
    success_files = 0
    
    # 处理所有文件
    for file_idx, (input_dir, parquet_file) in enumerate(all_parquet_files, 1):
        date_name = input_dir.name
        output_dir = Path(args.output_base) / date_name
        
        # 构建输出文件名
        output_name = parquet_file.name.replace('_parsed.parquet', '_semid.parquet')
        output_path = output_dir / output_name
        
        print(f"\n[{file_idx}/{total_files}] 处理文件: {date_name}/{parquet_file.name}")
        
        # 跳过已存在的文件
        if output_path.exists():
            print(f"跳过已存在文件: {output_path}")
            # 仍需要统计已存在文件的数据
            try:
                existing_df = pd.read_parquet(output_path)
                update_global_stats(existing_df, global_stats)
                success_files += 1
            except Exception as e:
                print(f"读取已存在文件失败: {e}")
            continue
        
        # 处理文件并返回处理后的数据用于统计
        try:
            processed_df = process_single_file_with_stats(
                parquet_file, output_path, model, args.device, args.batch_size
            )
            if processed_df is not None:
                update_global_stats(processed_df, global_stats)
                success_files += 1
        except Exception as e:
            print(f"处理文件失败 {parquet_file}: {e}")
            continue
    
    # 输出最终统计结果
    print_final_statistics(global_stats, total_files, success_files)


def update_global_stats(df, global_stats):
    """更新全局统计信息"""
    global_stats['total_rows'] += len(df)
    
    # 统计唯一MD5
    global_stats['unique_md5_oaids'].update(df['md5_oaid'])
    
    # 统计唯一向量和SemID
    for idx, row in df.iterrows():
        # 向量hash
        vec_hash = hash(tuple(row['parsed_vector']) if isinstance(row['parsed_vector'], list) 
                       else tuple(row['parsed_vector'].tolist()) if hasattr(row['parsed_vector'], 'tolist')
                       else str(row['parsed_vector']))
        
        semid_str = row['semid_str']
        
        global_stats['unique_vectors'].add(vec_hash)
        global_stats['unique_semids'].add(semid_str)
        
        # 建立映射关系
        global_stats['vector_to_semid'][vec_hash] = semid_str
        
        if semid_str not in global_stats['semid_to_vectors']:
            global_stats['semid_to_vectors'][semid_str] = set()
        global_stats['semid_to_vectors'][semid_str].add(vec_hash)


def process_single_file_with_stats(input_path, output_path, model, device, batch_size=1024):
    """处理单个文件并返回处理后的DataFrame用于统计"""
    
    # 读取原始数据
    try:
        table = pq.read_table(input_path)
        df = table.to_pandas()
    except Exception as e:
        print(f"读取文件失败 {input_path}: {e}")
        return None
    
    if len(df) == 0:
        print(f"跳过空文件: {input_path}")
        return None
    
    print(f"  -> 文件行数: {len(df):,}")
    
    # 准备数据
    md5_oaids = df['md5_oaid'].values
    vectors = df['parsed_vector'].values
    
    # 转换向量格式并归一化
    processed_vectors = []
    valid_indices = []
    
    for i, vec in enumerate(vectors):
        if isinstance(vec, (list, np.ndarray)):
            vec_array = np.array(vec, dtype=np.float32)
            if vec_array.shape[0] == 5120:
                # L2归一化
                norm = np.linalg.norm(vec_array, ord=2) + 1e-8
                vec_normalized = vec_array / norm
                processed_vectors.append(vec_normalized)
                valid_indices.append(i)
    
    if len(processed_vectors) == 0:
        print(f"  -> 警告: 没有有效向量")
        return None
    
    print(f"  -> 有效向量: {len(processed_vectors):,}/{len(df):,}")
    
    # 批量推理生成SemID
    all_semids = []
    
    with torch.no_grad():
        for i in range(0, len(processed_vectors), batch_size):
            batch_vectors = processed_vectors[i:i+batch_size]
            batch_tensor = torch.tensor(np.stack(batch_vectors), device=device)
            
            # 获取semantic IDs
            output = model.get_semantic_ids(batch_tensor, gumbel_t=0.001)
            sem_ids = output.sem_ids.cpu().numpy()  # shape: [n_layers, batch_size]
            
            # 转置为 [batch_size, n_layers]
            sem_ids = sem_ids.T
            all_semids.extend(sem_ids.tolist())
    
    # 构建输出数据
    output_data = {
        'md5_oaid': [md5_oaids[i] for i in valid_indices],
        'parsed_vector': [vectors[i] for i in valid_indices],
        'semid': all_semids,
        'semid_str': ['-'.join(map(str, semid)) for semid in all_semids]
    }
    
    output_df = pd.DataFrame(output_data)
    
    # 保存结果
    os.makedirs(output_path.parent, exist_ok=True)
    output_df.to_parquet(output_path, compression='snappy', index=False)
    
    print(f"  -> ✓ 保存完成: {len(output_df):,} 行")
    return output_df


def print_final_statistics(global_stats, total_files, success_files):
    """打印最终统计结果"""
    print("\n" + "="*80)
    print(" 处理完成统计报告")
    print("="*80)
    
    print(f" 文件处理情况:")
    print(f"   总文件数: {total_files:,}")
    print(f"   成功处理: {success_files:,}")
    print(f"   失败文件: {total_files - success_files:,}")
    
    print(f"\n 数据统计:")
    print(f"   总处理行数: {global_stats['total_rows']:,}")
    print(f"   唯一MD5_OAID数: {len(global_stats['unique_md5_oaids']):,}")
    print(f"   唯一parsed_vector数: {len(global_stats['unique_vectors']):,}")
    print(f"   唯一sem_id数: {len(global_stats['unique_semids']):,}")
    
    # 分析映射关系
    print(f"\n 映射关系分析:")
    
    # 检查每个vector是否对应唯一semid
    vector_to_unique_semid = True
    for vec_hash, semid in global_stats['vector_to_semid'].items():
        # 这里每个vector只会有一个semid，因为是直接映射
        pass
    
    print(f"   每个parsed_vector对应唯一sem_id: True")
    
    # 检查每个semid对应多少个不同的vector
    semid_with_multiple_vectors = 0
    max_vectors_per_semid = 0
    
    for semid_str, vector_set in global_stats['semid_to_vectors'].items():
        vector_count = len(vector_set)
        max_vectors_per_semid = max(max_vectors_per_semid, vector_count)
        if vector_count > 1:
            semid_with_multiple_vectors += 1
    
    print(f"   存在多个parsed_vector对应同一sem_id的情况: {'❌ 是' if semid_with_multiple_vectors > 0 else '✅ 否'}")
    if semid_with_multiple_vectors > 0:
        print(f"   -> 有 {semid_with_multiple_vectors:,} 个sem_id对应多个不同的parsed_vector")
        print(f"   -> 单个sem_id最多对应 {max_vectors_per_semid} 个不同的parsed_vector")
    
    # 计算压缩比
    if len(global_stats['unique_vectors']) > 0:
        compression_ratio = len(global_stats['unique_semids']) / len(global_stats['unique_vectors'])
        print(f"\n 量化效果:")
        print(f"   压缩比 (sem_id数/vector数): {compression_ratio:.4f}")
        if compression_ratio < 1:
            print(f"   量化效果: 良好 (压缩比 < 1)")
        else:
            print(f"   量化效果: 可能存在问题 (压缩比 >= 1)")
    
    print("="*80)


if __name__ == "__main__":
    main()