# train_rqvae.py
import gin
import os
import json
import re
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from scipy.stats import entropy
from collections import Counter
from typing import Dict, List, Tuple
import math
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

# 数据加载器
from data.custom_parquet_vectors import ParquetVectorIterable
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.utils import parse_config

torch._dynamo.disable()


class RQVAEEvaluator:
    """专门用于RQ-VAE第一阶段评估的工具类"""
    def __init__(self, n_layers: int, codebook_size: int):
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.history = {
            'steps': [],
            'codebook_usage': [[] for _ in range(n_layers)],
            'dead_codes': [[] for _ in range(n_layers)],
            'perplexity': [[] for _ in range(n_layers)],
            'gini_coefficient': [[] for _ in range(n_layers)],
            'collision_rate': [],
            'max_bucket_size': [],
            'eval_recon': [],
            'eval_total': []
        }

    @torch.no_grad()
    def evaluate_model(self, model, eval_dataloader, max_samples=10000, device='cuda') -> Dict:
        """完整评估RQ-VAE模型，返回所有关键指标"""
        model.eval()

        all_sem_ids = []
        eval_losses, eval_recon_losses, eval_vq_losses = [], [], []
        samples_processed = 0

        print(f"开始评估模型... 最多处理 {max_samples} 个样本")

        # 评估温度与训练最低温一致，避免极低温造成“看起来不用码本”的假象
        t_eval = 0.4

        eval_iter = iter(eval_dataloader)
        with tqdm(total=min(max_samples, 50000), desc="评估中", leave=False) as pbar:
            while samples_processed < max_samples:
                try:
                    batch = next(eval_iter)
                except StopIteration:
                    break

                mm = model.module if hasattr(model, "module") else model

                qout = mm.get_semantic_ids(batch, gumbel_t=t_eval)
                all_sem_ids.append(qout.sem_ids.cpu())  # [n_layers, batch]

                out = mm(batch, gumbel_t=t_eval)
                eval_losses.append(out.loss.item())
                eval_recon_losses.append(out.reconstruction_loss.item())
                eval_vq_losses.append(out.rqvae_loss.item())

                bsz = batch.shape[0]
                samples_processed += bsz
                pbar.update(bsz)

                if len(all_sem_ids) > 100:
                    break

        if not all_sem_ids:
            print("评估失败：没有获取到任何数据")
            return self._get_empty_metrics()

        # 合并所有semantic IDs: [n_layers, total_samples]
        all_sem_ids = torch.cat(all_sem_ids, dim=1)

        metrics = {
            'eval_loss': float(np.mean(eval_losses)),
            'eval_recon': float(np.mean(eval_recon_losses)),
            'eval_vq': float(np.mean(eval_vq_losses)),
            'samples_evaluated': samples_processed
        }

        # 各层指标
        for layer_idx in range(self.n_layers):
            layer_ids = all_sem_ids[layer_idx].numpy()
            metrics.update(self._compute_layer_metrics(layer_ids, layer_idx))

        # 碰撞指标
        metrics.update(self._compute_collision_metrics(all_sem_ids))
        return metrics

    def _compute_layer_metrics(self, layer_ids: np.ndarray, layer_idx: int) -> Dict:
        unique_ids, counts = np.unique(layer_ids, return_counts=True)

        used_codes = len(unique_ids)
        usage_rate = used_codes / self.codebook_size

        dead_codes = self.codebook_size - used_codes
        dead_code_rate = dead_codes / self.codebook_size

        probs = counts / counts.sum()
        H = entropy(probs, base=np.e)
        perplexity = np.exp(H)

        gini = self._compute_gini_coefficient(counts)

        return {
            f'layer_{layer_idx}_usage_rate': usage_rate,
            f'layer_{layer_idx}_used_codes': used_codes,
            f'layer_{layer_idx}_dead_codes': dead_codes,
            f'layer_{layer_idx}_dead_code_rate': dead_code_rate,
            f'layer_{layer_idx}_perplexity': perplexity,
            f'layer_{layer_idx}_entropy': H,
            f'layer_{layer_idx}_gini': gini,
            f'layer_{layer_idx}_max_count': counts.max(),
            f'layer_{layer_idx}_min_count': counts.min() if len(counts) > 0 else 0
        }

    def _compute_collision_metrics(self, all_sem_ids: torch.Tensor) -> Dict:
        """计算碰撞指标 - 前3层prefix的重复情况"""
        prefix_layers = min(self.n_layers - 1, 3)
        if prefix_layers <= 0:
            return {'collision_rate': 0.0, 'max_bucket_size': 1}

        total_samples = all_sem_ids.shape[1]
        sample_size = min(total_samples, 10000)
        indices = torch.randperm(total_samples)[:sample_size]

        prefixes = [tuple(all_sem_ids[:prefix_layers, i].tolist()) for i in indices]
        prefix_counts = Counter(prefixes)

        collision_buckets = sum(1 for c in prefix_counts.values() if c > 1)
        collision_rate = collision_buckets / len(prefix_counts) if len(prefix_counts) > 0 else 0.0
        max_bucket_size = max(prefix_counts.values()) if prefix_counts else 1

        return {
            'collision_rate': collision_rate,
            'max_bucket_size': max_bucket_size,
            'unique_prefixes': len(prefix_counts),
            'sampled_for_collision': len(prefixes)
        }

    def _compute_gini_coefficient(self, counts: np.ndarray) -> float:
        if len(counts) == 0:
            return 1.0
        counts = np.sort(counts)
        n = len(counts)
        cumsum = np.cumsum(counts)
        return 1 - 2 * np.sum(cumsum) / (n * cumsum[-1]) + 1 / n

    def _get_empty_metrics(self) -> Dict:
        metrics = {
            'eval_loss': 999.0,
            'eval_recon': 999.0,
            'samples_evaluated': 0,
            'collision_rate': 1.0,
            'max_bucket_size': 999,
            'unique_prefixes': 0,
            'sampled_for_collision': 0
        }
        for layer_idx in range(self.n_layers):
            metrics.update({
                f'layer_{layer_idx}_usage_rate': 0.0,
                f'layer_{layer_idx}_used_codes': 0,
                f'layer_{layer_idx}_dead_codes': self.codebook_size,
                f'layer_{layer_idx}_dead_code_rate': 1.0,
                f'layer_{layer_idx}_perplexity': 1.0,
                f'layer_{layer_idx}_entropy': 0.0,
                f'layer_{layer_idx}_gini': 1.0,
                f'layer_{layer_idx}_max_count': 0,
                f'layer_{layer_idx}_min_count': 0
            })
        return metrics

    def update_history(self, step: int, metrics: Dict):
        self.history['steps'].append(step)
        self.history['eval_recon'].append(metrics['eval_recon'])
        self.history['eval_total'].append(metrics['eval_loss'])
        self.history['collision_rate'].append(metrics['collision_rate'])
        self.history['max_bucket_size'].append(metrics['max_bucket_size'])

        for layer_idx in range(self.n_layers):
            self.history['codebook_usage'][layer_idx].append(
                metrics[f'layer_{layer_idx}_usage_rate']
            )
            self.history['dead_codes'][layer_idx].append(
                metrics[f'layer_{layer_idx}_dead_code_rate']
            )
            self.history['perplexity'][layer_idx].append(
                metrics[f'layer_{layer_idx}_perplexity']
            )
            self.history['gini_coefficient'][layer_idx].append(
                metrics[f'layer_{layer_idx}_gini']
            )

    def check_convergence(self, min_steps=15000, window_size=5,
                          usage_threshold=0.75, dead_code_threshold=0.08,
                          stability_threshold=0.01) -> Tuple[bool, str]:
        if len(self.history['steps']) < window_size:
            return False, f"评估次数不足，需要至少{window_size}次"

        current_step = self.history['steps'][-1]
        if current_step < min_steps:
            return False, f"训练步数不足，当前{current_step}，需要至少{min_steps}"

        recent_recon = self.history['eval_recon'][-window_size:]
        recon_cv = np.std(recent_recon) / np.mean(recent_recon)
        if recon_cv > stability_threshold:
            return False, f"重构损失不稳定，CV={recon_cv:.4f} > {stability_threshold}"

        for layer_idx in range(self.n_layers):
            recent_usage = self.history['codebook_usage'][layer_idx][-window_size:]
            recent_dead = self.history['dead_codes'][layer_idx][-window_size:]
            avg_usage = np.mean(recent_usage)
            avg_dead = np.mean(recent_dead)
            usage_var = np.var(recent_usage)
            if avg_usage < usage_threshold:
                return False, f"Layer {layer_idx} 使用率不足: {avg_usage:.3f} < {usage_threshold}"
            if avg_dead > dead_code_threshold:
                return False, f"Layer {layer_idx} 死亡码过多: {avg_dead:.3f} > {dead_code_threshold}"
            if usage_var > 0.02:
                return False, f"Layer {layer_idx} 使用率不稳定: var={usage_var:.4f}"

        return True, "收敛条件满足！"

    def should_save_as_best(self, current_metrics: Dict, best_metrics: Dict = None) -> bool:
        if best_metrics is None:
            return True
        cur_total = current_metrics['eval_loss']
        best_total = best_metrics['eval_loss']
        if cur_total < best_total * 0.999:
            return True
        if abs(cur_total - best_total) / max(best_total, 1e-12) < 0.001:
            cur_avg_usage = np.mean([current_metrics[f'layer_{i}_usage_rate'] for i in range(self.n_layers)])
            best_avg_usage = np.mean([best_metrics[f'layer_{i}_usage_rate'] for i in range(self.n_layers)])
            return cur_avg_usage > best_avg_usage
        return False

    def save_evaluation_summary(self, save_dir: str, metrics: Dict):
        os.makedirs(save_dir, exist_ok=True)
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_metrics[k] = float(v)
            else:
                clean_metrics[k] = v
        with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(clean_metrics, f, indent=2)
        self._plot_simple_summary(save_dir, metrics)
        print(f"✓ 评估摘要已保存到: {save_dir}")

    def _plot_simple_summary(self, save_dir: str, metrics: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        layer_indices = list(range(self.n_layers))

        ax = axes[0, 0]
        usage_rates = [metrics[f'layer_{i}_usage_rate'] for i in self.n_layers * [0]]
        usage_rates = [metrics[f'layer_{i}_usage_rate'] for i in range(self.n_layers)]
        ax.bar(layer_indices, usage_rates, color='skyblue', alpha=0.7)
        ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='目标阈值')
        ax.set_xlabel('Layer'); ax.set_ylabel('Usage Rate'); ax.set_title('各层码本使用率')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        perplexities = [metrics[f'layer_{i}_perplexity'] for i in range(self.n_layers)]
        ax.bar(layer_indices, perplexities, color='lightgreen', alpha=0.7)
        ax.set_xlabel('Layer'); ax.set_ylabel('Perplexity'); ax.set_title('各层困惑度')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        dead_rates = [metrics[f'layer_{i}_dead_code_rate'] for i in range(self.n_layers)]
        ax.bar(layer_indices, dead_rates, color='salmon', alpha=0.7)
        ax.axhline(y=0.08, color='red', linestyle='--', alpha=0.7, label='警戒线')
        ax.set_xlabel('Layer'); ax.set_ylabel('Dead Code Rate'); ax.set_title('各层死亡码比例')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.axis('off')
        avg_usage = np.mean(usage_rates)
        avg_dead = np.mean(dead_rates)
        avg_ppl = np.mean(perplexities)
        summary_text = f"""
关键指标摘要:

重构损失: {metrics['eval_recon']:.6f}
总损失: {metrics['eval_loss']:.6f}

平均使用率: {avg_usage:.3f}
平均死亡率: {avg_dead:.3f}
平均困惑度: {avg_ppl:.1f}

碰撞率: {metrics['collision_rate']:.3f}
最大桶大小: {metrics['max_bucket_size']}

评估样本数: {metrics['samples_evaluated']:,}
        """
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_summary.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


@gin.configurable
def train(
    # ==== 训练超参 ====
    iterations: int = 40000,
    batch_size: int = 1024,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    gradient_accumulate_every: int = 2,
    mixed_precision_type: str = "fp16",
    amp: bool = True,
    split_batches: bool = True,
    wandb_logging: bool = False,

    # ==== 路径 ====
    save_dir_root: str = "trained_models/rqvae_xieshen/",
    pretrained_rqvae_path: str = None,

    # ==== 数据 ====
    data_dirs=(
        "/mnt/xieshen/data/20250706",
        "/mnt/xieshen/data/20250707",
        "/mnt/xieshen/data/20250708",
        "/mnt/xieshen/data/20250709",
        "/mnt/xieshen/data/20250710",
        "/mnt/xieshen/data/20250711",
        "/mnt/xieshen/data/20250712",
    ),
    num_workers: int = 4,
    input_normalize: str = "l2",

    # ==== 保存/评估频率 ====
    save_model_every: int = 10000,
    do_eval: bool = True,
    eval_every: int = 3000,
    eval_samples: int = 8000,

    # ==== RQ-VAE 模型参数 ====
    commitment_weight: float = 0.25,
    vae_input_dim: int = 5120,
    vae_embed_dim: int = 512,
    vae_hidden_dims=[2048, 1024, 512],
    vae_codebook_size: int = 512,
    vae_codebook_normalize: bool = True,
    vae_codebook_mode=QuantizeForwardMode.ROTATION_TRICK,
    vae_sim_vq: bool = False,
    vae_n_layers: int = 3,
    vae_n_cat_feats: int = 0,
    vae_codebook_kmeans_init: bool = True,

    # ==== 收敛判据 ====
    convergence_min_steps: int = 15000,
    convergence_window_size: int = 5,
    convergence_usage_threshold: float = 0.75,
    convergence_dead_threshold: float = 0.08,
    convergence_stability_threshold: float = 0.01,
):
    # ==== Gumbel 温度调度：先高温探索，再退火（不低于 0.4） ====
    def gumbel_temp(it, total, warmup=5000, t_max=2.2, t_min=0.4):
        if it < warmup:
            t = t_max
        else:
            prog = (it - warmup) / max(1, total - warmup)
            t = t_min + (t_max - t_min) * 0.5 * (1.0 + math.cos(math.pi * prog))
        # 早期加入轻微 jitter，避免过早锁死到少量 code
        if it < 10000 and (it % 50 == 0):
            t = min(t * 1.25, 2.5)
        return float(t)

    if wandb_logging:
        params = locals()

    # ==== 初始化评估器 ====
    evaluator = RQVAEEvaluator(
        n_layers=vae_n_layers,
        codebook_size=vae_codebook_size
    )

    print(f"评估器初始化完成")
    print(f"收敛条件: 最少{convergence_min_steps}步，使用率≥{convergence_usage_threshold*100}%，死亡码≤{convergence_dead_threshold*100}%")
    print(f"数据规模: 2千万条记录，{vae_input_dim}维向量")

    # ==== 指标记录结构 ====
    metrics_history = {
        'steps': [], 'total_loss': [], 'recon_loss': [], 'vq_loss': [],
        'layer_usage': [[] for _ in range(vae_n_layers)],
        'layer_perplexity': [[] for _ in range(vae_n_layers)],
        'eval_loss': [], 'eval_recon': [], 'eval_vq': [], 'eval_steps': []
    }

    def ensure_list_lengths(metrics: dict):
        target_len = len(metrics['steps'])
        for key in ['layer_usage', 'layer_perplexity']:
            for i, lst in enumerate(metrics[key]):
                while len(lst) < target_len:
                    lst.append(lst[-1] if lst else 0)

    def save_training_visualizations(metrics, save_dir, config):
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.facecolor'] = 'white'
        fig = plt.figure(figsize=(16, 12))

        ax1 = plt.subplot(3, 3, 1)
        plt.plot(metrics['steps'], metrics['total_loss'], 'r-', label='Total Loss', linewidth=2)
        plt.plot(metrics['steps'], metrics['recon_loss'], 'b-', label='Recon Loss', linewidth=2)
        plt.xlabel('Steps'); plt.ylabel('Loss'); plt.title('Loss Evolution')
        plt.legend(); plt.grid(True, alpha=0.3)

        ax2 = plt.subplot(3, 3, 2)
        plt.plot(metrics['steps'], metrics['vq_loss'], 'g-', label='VQ Loss', linewidth=2)
        plt.xlabel('Steps'); plt.ylabel('VQ Loss'); plt.title('Quantization Loss')
        plt.legend(); plt.grid(True, alpha=0.3)

        ax3 = plt.subplot(3, 3, 3)
        colors = ['orange', 'purple', 'cyan', 'teal', 'olive', 'brown']
        for i in range(len(metrics['layer_usage'])):
            if metrics['layer_usage'][i]:
                steps_usage = metrics['steps'][:len(metrics['layer_usage'][i])]
                plt.plot(steps_usage, metrics['layer_usage'][i],
                         color=colors[i % len(colors)], label=f'Layer {i}', linewidth=2)
        plt.xlabel('Steps'); plt.ylabel('Used Codebook Entries'); plt.title('Codebook Usage per Layer')
        plt.legend(); plt.grid(True, alpha=0.3)

        ax4 = plt.subplot(3, 3, 4)
        for i in range(len(metrics['layer_perplexity'])):
            if metrics['layer_perplexity'][i]:
                steps_ppl = metrics['steps'][:len(metrics['layer_perplexity'][i])]
                plt.plot(steps_ppl, metrics['layer_perplexity'][i],
                         color=colors[i % len(colors)], label=f'Layer {i} PPL', linewidth=2)
        plt.xlabel('Steps'); plt.ylabel('Perplexity'); plt.title('Perplexity Evolution')
        plt.legend(); plt.grid(True, alpha=0.3)

        if metrics['eval_steps']:
            ax5 = plt.subplot(3, 3, 5)
            plt.plot(metrics['eval_steps'], metrics['eval_loss'], 'r--', label='Eval Total', marker='o')
            plt.plot(metrics['eval_steps'], metrics['eval_recon'], 'b--', label='Eval Recon', marker='s')
            plt.xlabel('Steps'); plt.ylabel('Evaluation Loss'); plt.title('Evaluation Metrics')
            plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        viz_path = os.path.join(save_dir, 'training_visualization.png')
        plt.savefig(viz_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        metrics_path = os.path.join(save_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ 训练可视化保存至: {viz_path}")
        print(f"✓ 训练数据保存至: {metrics_path}")

    # ===== Accelerator =====
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device

    print("=" * 60)
    print("RQ-VAE训练 - 大规模数据优化模式")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"批大小: {batch_size} (梯度累积: {gradient_accumulate_every})")
    print(f"有效批大小: {batch_size * gradient_accumulate_every}")
    print(f"学习率: {learning_rate}")

    # ===== 训练数据（自然分布）=====
    train_iterable = ParquetVectorIterable(
        data_dirs=list(data_dirs),
        column_vector="parsed_vector",
        column_id="md5_oaid",
        expected_dim=vae_input_dim,
        shuffle_files=True,
        shuffle_rows=True,
        seed=20250809,
        normalize=input_normalize,
        dtype="float32",
    )

    # Collate函数
    def collate(batch):
        _, vecs = zip(*batch)
        return torch.stack(vecs, dim=0)

    train_dataloader = DataLoader(
        train_iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # ===== 评估数据 =====
    eval_dataloader = None
    if do_eval:
        eval_iterable = ParquetVectorIterable(
            data_dirs=[data_dirs[-1]],
            column_vector="parsed_vector",
            column_id="md5_oaid",
            expected_dim=vae_input_dim,
            shuffle_files=False,
            shuffle_rows=True,
            seed=20250809,
            normalize=input_normalize,
            dtype="float32",
        )
        eval_dataloader = DataLoader(
            eval_iterable,
            batch_size=batch_size,
            num_workers=min(2, num_workers),
            collate_fn=collate,
            pin_memory=True,
        )

    # ===== 创建模型 =====
    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=list(vae_hidden_dims),
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=vae_codebook_kmeans_init,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ===== 创建优化器（码本单独组）=====
    codebook_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("layers") and "embedding.weight" in n:
            codebook_params.append(p)
        else:
            other_params.append(p)

    optimizer = AdamW(
        [
            {"params": other_params,   "lr": learning_rate,       "weight_decay": weight_decay},
            {"params": codebook_params,"lr": learning_rate * 1.0, "weight_decay": 0.0},
        ]
    )

    # ===== W&B =====
    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="rqvae-training-large-scale",
            config=params,
            name=f"large_L{vae_n_layers}_K{vae_codebook_size}_20M",
        )

    # ===== 断点恢复 =====
    start_iter = 0
    if pretrained_rqvae_path:
        checkpoint = torch.load(pretrained_rqvae_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iter"] + 1
        print(f"从迭代 {start_iter} 继续训练")

    # Accelerator准备
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    # ===== 训练循环 =====
    train_iter = iter(train_dataloader)
    losses = {"total": [], "recon": [], "vq": []}
    best_metrics = None

    os.makedirs(save_dir_root, exist_ok=True)

    with tqdm(initial=start_iter, total=iterations, disable=not accelerator.is_main_process) as pbar:
        for it in range(start_iter, iterations):
            model.train()

            # —— VQ 权重线性预热：0.5 → 3.0（前 10k 步） ——
            vq_warmup_steps = 10_000
            vq_w = 0.5 + (3.0 - 0.5) * min(1.0, it / vq_warmup_steps)
            mm = model.module if hasattr(model, "module") else model
            mm.set_loss_weights(vq_w=vq_w)

            # —— 首次 K-means 初始化 + 广播 ——
            if it == 0 and not pretrained_rqvae_path and vae_codebook_kmeans_init:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print("\n执行K-means初始化...")
                    model.eval()
                    init_data = []
                    for _ in range(200):
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_dataloader)
                            batch = next(train_iter)
                        init_data.append(batch)
                    init_data = torch.cat(init_data, dim=0)
                    print(f"K-means初始化数据: {init_data.shape}")
                    with torch.no_grad():
                        _ = model(init_data, gumbel_t=0.8)  # 用较高温初始化
                    model.train()
                    print("K-means初始化完成 (rank0)")
                accelerator.wait_for_everyone()
                mm = model.module if hasattr(model, "module") else model
                if dist.is_available() and dist.is_initialized():
                    with torch.no_grad():
                        for l, layer in enumerate(mm.layers):
                            if hasattr(layer, "embedding") and hasattr(layer.embedding, "weight"):
                                dist.broadcast(layer.embedding.weight.data, src=0)
                accelerator.wait_for_everyone()

            # —— 单步 —— 
            total_loss = 0.0
            recon_acc = 0.0
            vq_acc = 0.0
            optimizer.zero_grad(set_to_none=True)

            for _ in range(gradient_accumulate_every):
                try:
                    data = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    data = next(train_iter)

                with accelerator.autocast():
                    t = gumbel_temp(it, iterations)
                    out = model(data, gumbel_t=t)
                    loss = out.loss / gradient_accumulate_every
                    total_loss += loss

                    recon_acc += out.reconstruction_loss.detach().item() / gradient_accumulate_every
                    vq_acc    += out.rqvae_loss.detach().item() / gradient_accumulate_every

            accelerator.backward(total_loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # —— 死码轻量刷新（基于 residual 热启动） ——
            if (it % 2000 == 0) and it > 0:
                accelerator.wait_for_everyone()
                with torch.no_grad():
                    mm = model.module if hasattr(model, "module") else model
                    if accelerator.is_main_process:
                        try:
                            sample = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_dataloader)
                            sample = next(train_iter)

                        qout = mm.get_semantic_ids(sample, gumbel_t=1.0)
                        res_per_layer = qout.residuals
                        B = sample.size(0)

                        for l, layer in enumerate(mm.layers):
                            hist = torch.bincount(
                                qout.sem_ids[l].to(torch.int64),
                                minlength=layer.n_embed
                            ).float().to(mm.device)

                            thr = max(1, int(0.001 * B))
                            rare = (hist <= thr).nonzero(as_tuple=False).flatten()
                            if rare.numel() == 0:
                                continue

                            idx = torch.randint(0, B, (rare.numel(),), device=mm.device)
                            vecs = res_per_layer[l].T[idx]   # [num_rare, D]

                            if getattr(layer, "codebook_normalize", False) or (
                                hasattr(layer, "normalize") and layer.normalize
                            ):
                                vecs = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-6)

                            if hasattr(layer, "embedding") and hasattr(layer.embedding, "weight"):
                                layer.embedding.weight.data[rare] = vecs.to(layer.embedding.weight.data.dtype)

                    accelerator.wait_for_everyone()
                    mm = model.module if hasattr(model, "module") else model
                    if dist.is_available() and dist.is_initialized():
                        with torch.no_grad():
                            for l, layer in enumerate(mm.layers):
                                if hasattr(layer, "embedding") and hasattr(layer.embedding, "weight"):
                                    dist.broadcast(layer.embedding.weight.data, src=0)
                    accelerator.wait_for_everyone()

            # —— 记录损失 —— 
            losses["total"].append(float(total_loss.detach().item()))
            losses["recon"].append(float(recon_acc))
            losses["vq"].append(float(vq_acc))
            for k in losses:
                losses[k] = losses[k][-1000:]

            # —— 进度条与轻量指标 —— 
            if it % 100 == 0:
                avg_loss = float(np.mean(losses["total"]))
                avg_recon = float(np.mean(losses["recon"]))
                avg_vq = float(np.mean(losses["vq"]))

                metrics_history['steps'].append(it)
                metrics_history['total_loss'].append(avg_loss)
                metrics_history['recon_loss'].append(avg_recon)
                metrics_history['vq_loss'].append(avg_vq)

                try:
                    recent_logs = model.module.util_recent_logs if hasattr(model, 'module') else getattr(model, 'util_recent_logs', [])
                except Exception:
                    recent_logs = []
                if recent_logs:
                    latest_log = recent_logs[-1]
                    for layer_idx in range(vae_n_layers):
                        pattern = f"L{layer_idx}:(\\d+)/\\d+ [\\d.]+% ppl\\(b\\)=([\\d.]+)"
                        match = re.search(pattern, latest_log)
                        if match:
                            usage = int(match.group(1))
                            ppl = float(match.group(2))
                            metrics_history['layer_usage'][layer_idx].append(usage)
                            metrics_history['layer_perplexity'][layer_idx].append(ppl)

                ensure_list_lengths(metrics_history)
                pbar.set_description(
                    f"T:{gumbel_temp(it, iterations):.2f}|VQw:{vq_w:.2f}|L:{avg_loss:.6f}|R:{avg_recon:.6f}|VQ:{avg_vq:.6f}"
                )

            # —— W&B —— 
            if accelerator.is_main_process and wandb_logging and it % 100 == 0:
                log_dict = {
                    "train/loss": float(np.mean(losses["total"])),
                    "train/recon_loss": float(np.mean(losses["recon"])),
                    "train/vq_loss": float(np.mean(losses["vq"])),
                    "train/gumbel_t": gumbel_temp(it, iterations),
                    "train/vq_weight": vq_w,
                    "iter": it,
                }
                wandb.log(log_dict)

            # ======= 详细评估（仅主进程）=======
            if do_eval and (it + 1) % eval_every == 0 and eval_dataloader is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print(f"\n[详细评估开始] 迭代 {it}")
                    eval_metrics = evaluator.evaluate_model(
                        model=model,
                        eval_dataloader=eval_dataloader,
                        max_samples=eval_samples,
                        device=device
                    )
                    evaluator.update_history(it, eval_metrics)

                    print(f"[评估结果] 重构损失: {eval_metrics['eval_recon']:.6f}")
                    print(f"[评估结果] 总损失: {eval_metrics['eval_loss']:.6f}")
                    print(f"[评估结果] 样本数: {eval_metrics['samples_evaluated']:,}")

                    metrics_history['eval_steps'].append(it)
                    metrics_history['eval_loss'].append(eval_metrics['eval_loss'])
                    metrics_history['eval_recon'].append(eval_metrics['eval_recon'])
                    metrics_history['eval_vq'].append(eval_metrics.get('eval_vq', float('nan')))

                    avg_usage = 0
                    avg_dead = 0
                    for layer_idx in range(vae_n_layers):
                        usage = eval_metrics[f'layer_{layer_idx}_usage_rate']
                        dead = eval_metrics[f'layer_{layer_idx}_dead_code_rate']
                        ppl = eval_metrics[f'layer_{layer_idx}_perplexity']
                        print(f"  Layer {layer_idx}: 使用率={usage:.3f} 死亡率={dead:.3f} 困惑度={ppl:.1f}")
                        avg_usage += usage
                        avg_dead += dead
                    avg_usage /= vae_n_layers
                    avg_dead /= vae_n_layers
                    print(f"[平均指标] 使用率: {avg_usage:.3f}, 死亡率: {avg_dead:.3f}")
                    print(f"[碰撞统计] 碰撞率: {eval_metrics['collision_rate']:.3f}, 最大桶: {eval_metrics['max_bucket_size']}")

                    if evaluator.should_save_as_best(eval_metrics, best_metrics):
                        best_metrics = eval_metrics.copy()
                        os.makedirs(save_dir_root, exist_ok=True)
                        best_path = os.path.join(save_dir_root, "checkpoint_best.pt")
                        state = {
                            "iter": it,
                            "model": accelerator.get_state_dict(model),
                            "model_config": getattr(model, "config", {}),
                            "optimizer": optimizer.state_dict(),
                            "eval_metrics": eval_metrics,
                        }
                        torch.save(state, best_path)

                        eval_report_dir = os.path.join(save_dir_root, f"eval_report_step_{it}")
                        evaluator.save_evaluation_summary(eval_report_dir, eval_metrics)
                        print(f"✓ 新的最佳模型! 重构损失: {eval_metrics['eval_recon']:.6f}")

                    converged, conv_msg = evaluator.check_convergence(
                        min_steps=convergence_min_steps,
                        window_size=convergence_window_size,
                        usage_threshold=convergence_usage_threshold,
                        dead_code_threshold=convergence_dead_threshold,
                        stability_threshold=convergence_stability_threshold
                    )
                    if converged:
                        print(f"\n 模型已收敛! {conv_msg}")
                        print(f"可以停止训练并导出模型用于第二阶段。")
                    else:
                        print(f"⏳ 尚未收敛: {conv_msg}")

                    if wandb_logging:
                        wandb_dict = {
                            "eval/recon_loss": eval_metrics['eval_recon'],
                            "eval/total_loss": eval_metrics['eval_loss'],
                            "eval/collision_rate": eval_metrics['collision_rate'],
                            "eval/max_bucket_size": eval_metrics['max_bucket_size'],
                            "eval/avg_usage_rate": avg_usage,
                            "eval/avg_dead_rate": avg_dead,
                            "iter": it,
                        }
                        for layer_idx in range(vae_n_layers):
                            prefix = f"eval/layer_{layer_idx}"
                            wandb_dict.update({
                                f"{prefix}/usage_rate": eval_metrics[f'layer_{layer_idx}_usage_rate'],
                                f"{prefix}/dead_code_rate": eval_metrics[f'layer_{layer_idx}_dead_code_rate'],
                                f"{prefix}/perplexity": eval_metrics[f'layer_{layer_idx}_perplexity'],
                                f"{prefix}/gini": eval_metrics[f'layer_{layer_idx}_gini'],
                            })
                        wandb.log(wandb_dict)
                accelerator.wait_for_everyone()

            # ======= 定期保存 =======
            if accelerator.is_main_process and (it + 1) % save_model_every == 0:
                os.makedirs(save_dir_root, exist_ok=True)
                save_path = os.path.join(save_dir_root, f"checkpoint_{it}.pt")
                state = {
                    "iter": it,
                    "model": accelerator.get_state_dict(model),
                    "model_config": getattr(model, "config", {}),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, save_path)

                latest_path = os.path.join(save_dir_root, "checkpoint_latest.pt")
                if os.path.exists(latest_path):
                    try: os.remove(latest_path)
                    except FileNotFoundError: pass
                try:
                    os.symlink(f"checkpoint_{it}.pt", latest_path)
                except (FileExistsError, OSError):
                    import shutil
                    shutil.copy2(save_path, latest_path)

                print(f" 检查点已保存: checkpoint_{it}.pt")

            pbar.update(1)

    # ======= 最终保存 =======
    if accelerator.is_main_process:
        final_path = os.path.join(save_dir_root, "checkpoint_final.pt")
        state = {
            "iter": iterations - 1,
            "model": accelerator.get_state_dict(model),
            "model_config": getattr(model, "config", {}),
            "optimizer": optimizer.state_dict(),
        }
        os.makedirs(save_dir_root, exist_ok=True)
        torch.save(state, final_path)

        config_dict = {
            'vae_codebook_size': vae_codebook_size,
            'vae_n_layers': vae_n_layers,
            'commitment_weight': commitment_weight,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        save_training_visualizations(metrics_history, save_dir_root, config_dict)

        if best_metrics is not None:
            final_report_dir = os.path.join(save_dir_root, "final_evaluation_report")
            evaluator.save_evaluation_summary(final_report_dir, best_metrics)

            print(f"\n 最终评估报告:")
            print(f"  重构损失: {best_metrics['eval_recon']:.6f}")
            avg_usage = np.mean([best_metrics[f'layer_{i}_usage_rate'] for i in range(vae_n_layers)])
            avg_dead = np.mean([best_metrics[f'layer_{i}_dead_code_rate'] for i in range(vae_n_layers)])
            print(f"  平均使用率: {avg_usage:.3f}")
            print(f"  平均死亡率: {avg_dead:.3f}")
            print(f"  碰撞率: {best_metrics['collision_rate']:.3f}")

            quality_good = (
                avg_usage >= convergence_usage_threshold and
                avg_dead <= convergence_dead_threshold and
                best_metrics['collision_rate'] < 0.02
            )

            if quality_good:
                print(f" 模型质量良好，建议用于第二阶段训练！")
                print(f" Semantic ID生成质量指标:")
                print(f"   - 平均使用率: {avg_usage:.1%} ≥ {convergence_usage_threshold:.1%} ✓")
                print(f"   - 平均死亡率: {avg_dead:.1%} ≤ {convergence_dead_threshold:.1%} ✓")
                print(f"   - 碰撞率: {best_metrics['collision_rate']:.1%} < 2% ✓")
            else:
                print(f"  模型质量需要改进：")
                if avg_usage < convergence_usage_threshold:
                    print(f"   - 使用率偏低: {avg_usage:.3f} < {convergence_usage_threshold}")
                    print(f"     建议: 降低commitment_weight, 增加训练时间")
                if avg_dead > convergence_dead_threshold:
                    print(f"   - 死亡码过多: {avg_dead:.3f} > {convergence_dead_threshold}")
                    print(f"     建议: 检查k-means初始化, 调整commitment_weight")
                if best_metrics['collision_rate'] >= 0.02:
                    print(f"   - 碰撞率过高: {best_metrics['collision_rate']:.3f}")
                    print(f"     建议: 增加codebook_size或层数")

        print(f"\n 下一步操作建议:")
        print(f"  1. 检查最佳模型: {save_dir_root}/checkpoint_best.pt")
        print(f"  2. 查看评估报告: {save_dir_root}/final_evaluation_report/")
        print(f"  3. 如质量满意，可用此模型开始第二阶段 Seq2Seq 训练")
        print(f"  4. 大数据集建议: 可以适当放宽标准，75%使用率也可接受")

        model_config_path = os.path.join(save_dir_root, "model_config_for_stage2.json")

        clean_metrics = {}
        if best_metrics:
            for k, v in best_metrics.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean_metrics[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean_metrics[k] = v.tolist()
                else:
                    clean_metrics[k] = v

        stage2_config = {
            "rqvae_config": {
                "input_dim": vae_input_dim,
                "embed_dim": vae_embed_dim,
                "hidden_dims": vae_hidden_dims,
                "codebook_size": vae_codebook_size,
                "n_layers": vae_n_layers,
                "commitment_weight": commitment_weight,
            },
            "semantic_id_length": vae_n_layers + 1,  # +1 for collision suffix
            "vocab_size_estimate": vae_codebook_size * vae_n_layers + 1000,
            "best_checkpoint": "checkpoint_best.pt",
            "data_scale": f"vectors_{vae_input_dim}dim",
            "quality_metrics": clean_metrics,
        }

        with open(model_config_path, 'w') as f:
            json.dump(stage2_config, f, indent=2)
        print(f"  5. 第二阶段配置已保存: {model_config_path}")

        print("\nRQ-VAE第一阶段训练完成")

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
