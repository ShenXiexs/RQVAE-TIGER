# dataset_loader.py
"""
步骤3: 数据集加载器（逐 code 自回归版）
读取 *_samples_tokens.pkl（history_tokens, target_tokens），并构造
encoder_input_ids / encoder_attention_mask / decoder_input_ids / labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import numpy as np
from typing import Dict, Tuple
import argparse
from pathlib import Path


# -------------------- 工具 --------------------

def load_vocab(vocab_path: str) -> Dict:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    special = vocab["special_tokens"]  # {"<PAD>": id, "<BOS>": id, "<EOS>": id}
    user_bucket_tokens = vocab["user_bucket_tokens"]  # list of ids
    dim_value_to_token = vocab["dim_value_to_token"]  # list[ dict(str(value) -> id) ]
    # 计算词表大小
    vocab_size = len(special) + len(user_bucket_tokens)
    for d in dim_value_to_token:
        vocab_size += len(d)
    code_dims = vocab["code_dims"]
    return {
        "pad_id": special["<PAD>"],
        "bos_id": special["<BOS>"],
        "eos_id": special["<EOS>"],
        "vocab_size": vocab_size,
        "code_dims": code_dims
    }


# -------------------- Dataset --------------------

class TokenLevelSeqDataset(Dataset):
    """
    样本来自 *_samples_tokens.pkl：
      - history_tokens: List[int]  扁平 token（含用户桶 + 若干item的m个code）
      - target_tokens:  List[int]  下一个item的 m 个 code token
    返回：
      - encoder_input_ids         [B, L_enc]
      - encoder_attention_mask    [B, L_enc]   (bool)
      - decoder_input_ids         [B, L_dec]   (BOS + target_tokens + PAD)
      - labels                    [B, L_dec]   (target_tokens + EOS + -100 for PAD)
    """
    def __init__(
        self,
        samples_file: str,
        vocab_info: Dict,
        max_seq_tokens: int = 64,
        label_ignore_index: int = -100
    ):
        self.max_seq_tokens = max_seq_tokens
        self.pad_id = vocab_info["pad_id"]
        self.bos_id = vocab_info["bos_id"]
        self.eos_id = vocab_info["eos_id"]
        self.code_dims = vocab_info["code_dims"]
        self.label_ignore_index = label_ignore_index

        with open(samples_file, "rb") as f:
            self.samples = pickle.load(f)

        print(f"已加载 {len(self.samples):,} 个样本，来自文件：{samples_file}")
        # 简要统计
        h_lens = [len(s["history_tokens"]) for s in self.samples]
        print(f"历史序列token数: 最小={min(h_lens)}, 最大={max(h_lens)}, 平均={np.mean(h_lens):.2f}")
        t_lens = [len(s["target_tokens"]) for s in self.samples]
        print(f"目标序列token数: 最小={min(t_lens)}, 最大={max(t_lens)}, 平均={np.mean(t_lens):.2f} (预期={self.code_dims})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sp = self.samples[idx]
        history = list(sp["history_tokens"])
        target  = list(sp["target_tokens"])[: self.code_dims]  # 保险：截到 m

        # --- Encoder side ---
        if len(history) > self.max_seq_tokens:
            history = history[-self.max_seq_tokens:]

        enc_len = len(history)
        encoder_input_ids = history + [self.pad_id] * (self.max_seq_tokens - enc_len)
        encoder_attention_mask = [1] * enc_len + [0] * (self.max_seq_tokens - enc_len)

        # --- Decoder side ---
        # decoder_input = [BOS] + target
        # labels        = target + [EOS]
        dec_inp = [self.bos_id] + target
        labels  = target + [self.eos_id]

        dec_len = len(dec_inp)  # 通常 = m+1
        # 动态长度：让 collate 做pad（更高效），这里先返回变长序列
        return {
            "encoder_input_ids": torch.tensor(encoder_input_ids, dtype=torch.long),
            "encoder_attention_mask": torch.tensor(encoder_attention_mask, dtype=torch.bool),
            "decoder_input_ids": torch.tensor(dec_inp, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "enc_len": torch.tensor(enc_len, dtype=torch.long),
            "dec_len": torch.tensor(dec_len, dtype=torch.long),
        }


# -------------------- Collator（对 decoder 侧做动态 padding） --------------------

class TokenLevelCollator:
    def __init__(self, pad_id: int, label_ignore_index: int = -100):
        self.pad_id = pad_id
        self.label_ignore_index = label_ignore_index

    def __call__(self, batch):
        # encoder 部分已经是固定长度（max_seq_tokens），直接 stack
        enc_input = torch.stack([b["encoder_input_ids"] for b in batch])
        enc_mask  = torch.stack([b["encoder_attention_mask"] for b in batch])

        # decoder 动态 pad 到该 batch 的最大长度
        max_dec = max(b["decoder_input_ids"].size(0) for b in batch)
        dec_input_list = []
        labels_list = []
        for b in batch:
            d = b["decoder_input_ids"]
            l = b["labels"]
            pad_len = max_dec - d.size(0)
            if pad_len > 0:
                d = torch.cat([d, torch.full((pad_len,), self.pad_id, dtype=torch.long)])
                # labels 对 pad 位置用 ignore index
                l = torch.cat([l, torch.full((pad_len,), self.label_ignore_index, dtype=torch.long)])
            dec_input_list.append(d)
            labels_list.append(l)

        dec_input = torch.stack(dec_input_list)
        labels    = torch.stack(labels_list)

        return {
            "encoder_input_ids": enc_input,         # [B, L_enc]
            "encoder_attention_mask": enc_mask,     # [B, L_enc]
            "decoder_input_ids": dec_input,         # [B, L_dec]
            "labels": labels,                       # [B, L_dec]  (-100 on pads)
        }


# -------------------- Loader 创建 --------------------

def create_data_loaders(
    data_dir: str,
    batch_size: int = 256,
    max_seq_tokens: int = 64,
    num_workers: int = 4,
    seed: int = 20250815,  # 添加种子参数
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    data_dir = Path(data_dir)
    vocab = load_vocab(str(data_dir / "vocab.json"))
    print(f"词表大小: {vocab['vocab_size']:,}, 代码维度: {vocab['code_dims']}")

    # 创建随机生成器（确保数据加载的可复现性）
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    loaders = {}
    for split in ["train", "validation", "test"]:
        samples_file = data_dir / f"{split}_samples_tokens.pkl"
        if not samples_file.exists():
            print(f"警告: 文件 {samples_file} 不存在，跳过 {split} 集")
            continue

        ds = TokenLevelSeqDataset(
            samples_file=str(samples_file),
            vocab_info=vocab,
            max_seq_tokens=max_seq_tokens,
            label_ignore_index=-100
        )
        collator = TokenLevelCollator(pad_id=vocab["pad_id"], label_ignore_index=-100)
        
        # 只有训练集需要 shuffle 和 generator
        if split == "train":
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True,
                drop_last=True,
                generator=generator,  # 添加生成器以确保shuffle的可复现性
            )
        else:
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True,
                drop_last=False,
            )
        
        loaders[split] = loader
        
        split_name = {"train": "训练", "validation": "验证", "test": "测试"}[split]
        print(f"{split_name}集加载器: {len(ds):,} 个样本, {len(loader):,} 个批次")

    metadata = {
        "vocab_size": vocab["vocab_size"],
        "pad_token": vocab["pad_id"],
        "bos_token": vocab["bos_id"],
        "eos_token": vocab["eos_id"],
        "code_dims": vocab["code_dims"],
        "max_seq_tokens": max_seq_tokens,
    }
    return loaders.get("train"), loaders.get("validation"), loaders.get("test"), metadata


# -------------------- 简单自测 --------------------

def main():
    parser = argparse.ArgumentParser(description="测试Token级数据加载器")
    parser.add_argument("--data_dir", default="./sequence_data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_seq_tokens", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250815, help="随机种子")
    args = parser.parse_args()

    # 设置种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("Token级数据加载器测试")
    print(f"随机种子: {args.seed}")
    print("=" * 80)

    train_loader, val_loader, test_loader, meta = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_tokens=args.max_seq_tokens,
        num_workers=args.num_workers,
        seed=args.seed
    )

    if train_loader is None:
        print("错误：未找到训练集加载器，请检查数据文件。")
        return

    for i, batch in enumerate(train_loader):
        print(f"\n批次 {i+1}:")
        print("  编码器输入ID形状:", batch["encoder_input_ids"].shape)
        print("  编码器注意力掩码形状:", batch["encoder_attention_mask"].shape)
        print("  解码器输入ID形状:", batch["decoder_input_ids"].shape)
        print("  标签形状:", batch["labels"].shape)
        if i + 1 >= args.test_batches:
            break

    print("\n元数据:", meta)
    print("测试完成！")


if __name__ == "__main__":
    main()