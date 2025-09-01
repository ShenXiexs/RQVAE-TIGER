#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复后的TIGER模型评估
- 使用Top-K采样生成多个候选
- 或者使用束搜索（高效版本）
- 确保指标能正确区分
"""

import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 本地模块
from dataset_loader import create_data_loaders
from tiger_model import TIGERSeq2SeqRecommender


def load_vocab(vocab_path: str):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    special = vocab["special_tokens"]
    user_bucket_tokens = vocab["user_bucket_tokens"]
    dim_value_to_token = vocab["dim_value_to_token"]
    code_dims = int(vocab["code_dims"])

    vocab_size = len(special) + len(user_bucket_tokens)
    for d in dim_value_to_token:
        vocab_size += len(d)

    token_to_code_value = []
    dim_token_id_sets = []
    for d in dim_value_to_token:
        inv = {int(tok): int(val) for val, tok in d.items()}
        token_to_code_value.append(inv)
        dim_token_id_sets.append(set(inv.keys()))

    info = {
        "pad_id": special["<PAD>"],
        "bos_id": special["<BOS>"],
        "eos_id": special["<EOS>"],
        "vocab_size": vocab_size,
        "code_dims": code_dims,
        "token_to_code_value": token_to_code_value,
        "dim_token_id_sets": dim_token_id_sets,
    }
    return info


def tokens_to_semid(token_seq, token_to_code_value):
    codes = []
    for d, tok in enumerate(token_seq):
        v = token_to_code_value[d].get(int(tok))
        if v is None:
            return None
        codes.append(v)
    return tuple(codes)


# -------------------- 高效的束搜索 --------------------

def efficient_beam_search(
    model,
    encoder_input_ids,
    encoder_attention_mask,
    bos_id: int,
    code_dims: int,
    dim_token_id_sets,
    beam_size: int = 10,
    device: torch.device = torch.device("cuda"),
    amp: bool = False,
):
    """
    高效的束搜索：每个step扩展所有beam，然后选择top-K
    """
    model.eval()
    B = encoder_input_ids.size(0)
    
    # 初始化：每个样本有一个beam [BOS]
    # beams: List[List[Tuple[tokens, score]]] 长度为B
    beams = [[(torch.tensor([bos_id], device=device), 0.0)] for _ in range(B)]
    
    with torch.inference_mode():
        for step in range(code_dims):
            new_beams = [[] for _ in range(B)]
            
            for batch_idx in range(B):
                candidates = []
                
                for beam_tokens, beam_score in beams[batch_idx]:
                    # 为当前beam生成下一个token的候选
                    dec_input = beam_tokens.unsqueeze(0)  # [1, seq_len]
                    
                    if amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(
                                encoder_input_ids=encoder_input_ids[batch_idx:batch_idx+1],
                                encoder_attention_mask=encoder_attention_mask[batch_idx:batch_idx+1],
                                decoder_input_ids=dec_input,
                            )
                    else:
                        outputs = model(
                            encoder_input_ids=encoder_input_ids[batch_idx:batch_idx+1],
                            encoder_attention_mask=encoder_attention_mask[batch_idx:batch_idx+1],
                            decoder_input_ids=dec_input,
                        )
                    
                    logits = outputs["logits"][0, -1, :]  # [vocab_size]
                    
                    # 维度掩码
                    if step < len(dim_token_id_sets):
                        allowed_tokens = list(dim_token_id_sets[step])
                        mask = torch.full_like(logits, float("-inf"))
                        mask[allowed_tokens] = 0.0
                        logits = logits + mask
                    
                    # 计算log概率
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # 取top-K候选
                    topk_scores, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)
                    
                    for k in range(beam_size):
                        new_tokens = torch.cat([beam_tokens, topk_tokens[k:k+1]])
                        new_score = beam_score + topk_scores[k].item()
                        candidates.append((new_tokens, new_score))
                
                # 从所有候选中选择top beam_size
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beams[batch_idx] = candidates[:beam_size]
            
            beams = new_beams
    
    # 整理结果：去掉BOS，只保留tokens
    results = []
    for batch_idx in range(B):
        batch_results = []
        for tokens, score in beams[batch_idx]:
            token_list = tokens[1:].cpu().tolist()  # 去掉BOS
            batch_results.append((token_list, score))
        results.append(batch_results)
    
    return results


# -------------------- 更简单的Top-P采样 --------------------

def nucleus_sampling_batch(
    model,
    encoder_input_ids,
    encoder_attention_mask,
    bos_id: int,
    code_dims: int,
    dim_token_id_sets,
    num_samples: int = 10,
    top_p: float = 0.9,
    temperature: float = 1.0,
    device: torch.device = torch.device("cuda"),
    amp: bool = False,
):
    """
    批量核采样 (Top-P)，生成多样化的候选
    """
    model.eval()
    B = encoder_input_ids.size(0)
    all_candidates = [[] for _ in range(B)]
    
    with torch.inference_mode():
        for sample_idx in range(num_samples):
            decoder_input = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
            
            for step in range(code_dims):
                if amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(
                            encoder_input_ids=encoder_input_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            decoder_input_ids=decoder_input,
                        )
                else:
                    outputs = model(
                        encoder_input_ids=encoder_input_ids,
                        encoder_attention_mask=encoder_attention_mask,
                        decoder_input_ids=decoder_input,
                    )
                
                logits = outputs["logits"][:, -1, :] / temperature  # [B, V]
                
                # 维度掩码
                if step < len(dim_token_id_sets):
                    allowed_tokens = list(dim_token_id_sets[step])
                    mask = torch.full_like(logits, float("-inf"))
                    mask[:, allowed_tokens] = 0.0
                    logits = logits + mask
                
                # Top-P (nucleus) 采样
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 找到累积概率超过top_p的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 构建最终的采样分布
                final_probs = probs.clone()
                for b in range(B):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    final_probs[b, indices_to_remove] = 0.0
                    final_probs[b] = final_probs[b] / final_probs[b].sum()  # 重新归一化
                
                # 采样
                next_tokens = torch.multinomial(final_probs, 1)  # [B, 1]
                decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            
            # 存储结果（去掉BOS）
            sequences = decoder_input[:, 1:].cpu().numpy()
            for b in range(B):
                all_candidates[b].append(sequences[b].tolist())
    
    return all_candidates


# -------------------- 指标计算 --------------------

def recall_at_k(pred_semids, true_semids, k: int):
    hits = []
    for preds, tgt in zip(pred_semids, true_semids):
        if tgt is None:
            continue
        topk = preds[:k] if len(preds) >= k else preds
        hits.append(1.0 if tgt in topk else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def ndcg_at_k(pred_semids, true_semids, k: int):
    vals = []
    for preds, tgt in zip(pred_semids, true_semids):
        if tgt is None:
            continue
        topk = preds[:k] if len(preds) >= k else preds
        ndcg = 0.0
        for rank, s in enumerate(topk):
            if s == tgt:
                ndcg = 1.0 / math.log2(rank + 2)
                break
        vals.append(ndcg)
    return float(np.mean(vals)) if vals else 0.0


def mrr(pred_semids, true_semids):
    vals = []
    for preds, tgt in zip(pred_semids, true_semids):
        if tgt is None:
            continue
        rr = 0.0
        for rank, s in enumerate(preds):
            if s == tgt:
                rr = 1.0 / (rank + 1)
                break
        vals.append(rr)
    return float(np.mean(vals)) if vals else 0.0


def coverage(pred_semids, universe_count: int):
    uniq = set()
    for preds in pred_semids:
        uniq.update(preds)
    return len(uniq) / max(universe_count, 1)


# -------------------- 评估器 --------------------

class TIGEREvaluator:
    def __init__(self, model_path: str, data_dir: str, device: str = "auto",
                 generation_mode: str = "beam", beam_size: int = 10,
                 num_samples: int = 10, top_p: float = 0.9, 
                 temperature: float = 1.0, amp: bool = False):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                                   else (device if device != "auto" else "cpu"))
        self.generation_mode = generation_mode  # "beam", "nucleus", "greedy"
        self.beam_size = beam_size
        self.num_samples = num_samples
        self.top_p = top_p
        self.temperature = temperature
        self.amp = amp
        
        self.model = None
        self.test_loader = None
        self.vocab = None
        self.token_to_code_value = None
        self.dim_token_id_sets = None
        self.code_dims = None
        self.bos_id = None
        self.semid_universe = None

        self._load_vocab_and_universe()
        self._load_model_and_data()

    def _load_vocab_and_universe(self):
        vocab_path = self.data_dir / "vocab.json"
        self.vocab = load_vocab(str(vocab_path))
        self.bos_id = self.vocab["bos_id"]
        self.code_dims = self.vocab["code_dims"]
        self.token_to_code_value = self.vocab["token_to_code_value"]
        self.dim_token_id_sets = self.vocab["dim_token_id_sets"]

        semid2items_path = self.data_dir / "semid_to_items.json"
        if semid2items_path.exists():
            with open(semid2items_path, "r") as f:
                semid2items = json.load(f)
            self.semid_universe = len(semid2items)
        else:
            self.semid_universe = 0

    def _load_model_and_data(self):
        print("加载模型和测试数据...")
        
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        vocab_size = self.vocab["vocab_size"]

        self.model = TIGERSeq2SeqRecommender(
            vocab_size=vocab_size,
            d_model=config.get("d_model", 512),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 6),
            max_seq_length=config.get("max_seq_tokens", 64),
            pad_token_id=self.vocab["pad_id"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        _, _, self.test_loader, _ = create_data_loaders(
            data_dir=str(self.data_dir),
            batch_size=config.get("batch_size", 256),
            max_seq_tokens=config.get("max_seq_tokens", 64),
            num_workers=4,
        )
        
        print(f"模型已加载。参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"测试批次数: {len(self.test_loader):,}")
        print(f"生成模式: {self.generation_mode}")

    def evaluate(self, k_values=(5, 10), limit_batches: int = 0):
        print("在测试集上进行评估（语义ID级别）...")
        all_loss = []
        all_pred_semids = []
        all_true_semids = []

        with torch.inference_mode():
            for bi, batch in enumerate(tqdm(self.test_loader, desc="评估中")):
                enc_inp = batch["encoder_input_ids"].to(self.device)
                enc_mask = batch["encoder_attention_mask"].to(self.device)
                dec_inp = batch["decoder_input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 计算损失
                if self.amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(
                            encoder_input_ids=enc_inp,
                            encoder_attention_mask=enc_mask,
                            decoder_input_ids=dec_inp,
                        )["logits"]
                else:
                    logits = self.model(
                        encoder_input_ids=enc_inp,
                        encoder_attention_mask=enc_mask,
                        decoder_input_ids=dec_inp,
                    )["logits"]

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
                all_loss.append(loss.item())

                # 生成候选
                if self.generation_mode == "beam":
                    batch_candidates = efficient_beam_search(
                        model=self.model,
                        encoder_input_ids=enc_inp,
                        encoder_attention_mask=enc_mask,
                        bos_id=self.bos_id,
                        code_dims=self.code_dims,
                        dim_token_id_sets=self.dim_token_id_sets,
                        beam_size=self.beam_size,
                        device=self.device,
                        amp=self.amp,
                    )
                    # 转换格式：只保留tokens
                    batch_candidates = [[tokens for tokens, score in candidates] 
                                       for candidates in batch_candidates]
                
                elif self.generation_mode == "nucleus":
                    batch_candidates = nucleus_sampling_batch(
                        model=self.model,
                        encoder_input_ids=enc_inp,
                        encoder_attention_mask=enc_mask,
                        bos_id=self.bos_id,
                        code_dims=self.code_dims,
                        dim_token_id_sets=self.dim_token_id_sets,
                        num_samples=self.num_samples,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        device=self.device,
                        amp=self.amp,
                    )

                # 转换为semid
                batch_pred_semids = []
                for candidates in batch_candidates:
                    semids = []
                    for tokens in candidates:
                        semid = tokens_to_semid(tokens, self.token_to_code_value)
                        if semid is not None:
                            semids.append(semid)
                    # 去重但保持顺序
                    seen = set()
                    unique_semids = []
                    for s in semids:
                        if s not in seen:
                            seen.add(s)
                            unique_semids.append(s)
                    batch_pred_semids.append(unique_semids)
                
                all_pred_semids.extend(batch_pred_semids)

                # 处理真实semid
                labels_cpu = labels.detach().cpu().numpy()
                for i in range(labels_cpu.shape[0]):
                    tgt_tokens = labels_cpu[i][:self.code_dims]
                    tgt_tokens = [t for t in tgt_tokens if t not in [-100, self.bos_id, self.vocab["eos_id"]]]
                    tgt_tokens = tgt_tokens[:self.code_dims]
                    
                    if len(tgt_tokens) == self.code_dims:
                        semid = tokens_to_semid(tgt_tokens, self.token_to_code_value)
                        all_true_semids.append(semid)
                    else:
                        all_true_semids.append(None)

                if limit_batches > 0 and (bi + 1) >= limit_batches:
                    break

        # 计算指标
        metrics = {"测试损失": float(np.mean(all_loss)) if all_loss else 0.0}
        for k in k_values:
            metrics[f"召回率@{k}"] = recall_at_k(all_pred_semids, all_true_semids, k)
            metrics[f"NDCG@{k}"] = ndcg_at_k(all_pred_semids, all_true_semids, k)
        metrics["MRR"] = mrr(all_pred_semids, all_true_semids)
        metrics["覆盖率"] = coverage(all_pred_semids, self.semid_universe)

        return metrics, all_pred_semids, all_true_semids

    def save_report(self, metrics, output_dir: str, k_values=(5, 10)):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "evaluation_report.json", "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"评估报告已保存到 {out}")


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="评估 TIGER 模型（修复版）")
    parser.add_argument("--model_path", required=True, help="模型检查点路径")
    parser.add_argument("--data_dir", required=True, help="数据目录")
    parser.add_argument("--output_dir", default="./evaluation_results", help="输出目录")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--generation_mode", choices=["beam", "nucleus", "greedy"], default="beam", 
                        help="生成模式")
    parser.add_argument("--beam_size", type=int, default=10, help="束搜索大小")
    parser.add_argument("--num_samples", type=int, default=10, help="采样数量")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus采样参数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--amp", action="store_true", help="使用混合精度")
    parser.add_argument("--limit_batches", type=int, default=0, help="限制批次数量（调试用）")
    args = parser.parse_args()

    print("=" * 80)
    print("TIGER 模型评估（修复版）")
    print("=" * 80)

    evaluator = TIGEREvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
        generation_mode=args.generation_mode,
        beam_size=args.beam_size,
        num_samples=args.num_samples,
        top_p=args.top_p,
        temperature=args.temperature,
        amp=args.amp,
    )
    
    metrics, _, _ = evaluator.evaluate(k_values=tuple(args.k_values), limit_batches=args.limit_batches)
    evaluator.save_report(metrics, args.output_dir, k_values=tuple(args.k_values))

    print("\n评估结果摘要:")
    print(f"  测试损失: {metrics.get('测试损失', 0):.4f}")
    for k in args.k_values:
        print(f"  召回率@{k}: {metrics.get(f'召回率@{k}', 0):.4f}")
        print(f"  NDCG@{k}: {metrics.get(f'NDCG@{k}', 0):.4f}")
    print(f"  MRR: {metrics.get('MRR', 0):.4f}")
    print(f"  覆盖率: {metrics.get('覆盖率', 0):.4f}")
    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()