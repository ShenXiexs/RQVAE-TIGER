#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
步骤2: 构建用户序列（仅使用“已包含 semid 的数据集”）
- 输入目录(仅 semid_data_dirs)：每个 parquet 至少包含 md5_oaid, semid（以及你的原始列）
- 逐 code 自回归：把 semid 的每一维视为一个 token，训练目标=下一个 item 的 m 个 code token
- 输出：
  1) train/validation/test_sequences_tokens.pkl（user -> 扁平token序列）
  2) train/validation/test_samples_tokens.pkl（样本：history_tokens -> target_tokens[m]）
  3) vocab.json（special/user/code token 映射）
  4) semid_to_items.json（semid字符串 -> 本ID对应的“item键”集合；这里用行位置信息充当占位）
  5) mappings.json（维度与取值规模等）
"""

import os
import json
import time
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def _row_item_key(date_name: str, file_idx: int, row_idx: int) -> str:
    return f"{date_name}_{file_idx:03d}_{int(row_idx):06d}"


class SequenceBuilder:
    def __init__(
        self,
        semid_data_dirs,
        output_dir,
        min_items=5,            # 用户最少交互 item 数（小于该阈值丢弃）
        max_items=10,           # 历史窗口的最大 item 数（仅用于构造样本的历史，None/<=0 表示不限）
        user_bucket_count=2000, # 用户桶数量
        max_seq_tokens=64,     # 扁平 token 保护上限
        expected_code_dims=4    # 期望的 semid 维度（不一致时以数据为准）
    ):
        self.semid_data_dirs = list(semid_data_dirs)
        self.output_dir = output_dir
        self.min_items = int(min_items)
        self.max_items = None if (max_items is None or int(max_items) <= 0) else int(max_items)
        self.user_bucket_count = int(user_bucket_count)
        self.max_seq_tokens = int(max_seq_tokens)
        self.expected_code_dims = int(expected_code_dims)

        os.makedirs(self.output_dir, exist_ok=True)

        # 词表映射
        self.code_value_to_token = []
        self.token_to_code_value = []
        self.special_tokens = {}
        self.user_bucket_tokens = []

        # 维度统计
        self.code_dims = None
        self.dim_unique_values = []

        # 评估辅助：semid -> item 键集合
        self.semid_to_items = defaultdict(set)

    # ---------------- 文件扫描 ----------------

    def _scan_parquets(self, d: str):
        if not os.path.exists(d):
            return []
        pats = ["_parsed.parquet", "_semid.parquet", ".parquet"]
        for suffix in pats:
            files = sorted([f for f in os.listdir(d) if f.endswith(suffix)])
            if files:
                return [os.path.join(d, f) for f in files]
        return []

    # ---------------- 载入 semid 并构建 vocab ----------------

    def load_semid_and_build_vocab(self):
        print("【1/5】读取带 semid 的数据并构建词表...")
        total_rows = 0

        for semid_dir in tqdm(self.semid_data_dirs, desc="扫描目录"):
            date_name = os.path.basename(semid_dir.rstrip("/"))
            files = self._scan_parquets(semid_dir)
            for file_idx, fp in enumerate(tqdm(files, desc=date_name, leave=False)):
                try:
                    df = pd.read_parquet(fp)
                except Exception as e:
                    print(f"[跳过] 读取失败: {fp} ({e})")
                    continue

                if "md5_oaid" not in df.columns or "semid" not in df.columns:
                    print(f"[跳过] 缺少必要列(md5_oaid/semid): {fp}")
                    continue

                for row_idx, row in df.iterrows():
                    total_rows += 1
                    semid_raw = row["semid"]

                    if isinstance(semid_raw, (list, tuple, np.ndarray)):
                        semid = tuple(int(x) for x in semid_raw)
                    else:
                        try:
                            s = str(semid_raw).strip().strip("[]")
                            semid = tuple(int(x) for x in s.split(",") if str(x).strip() != "")
                        except Exception:
                            continue

                    if self.code_dims is None:
                        self.code_dims = len(semid)
                        self.dim_unique_values = [set() for _ in range(self.code_dims)]

                    if len(semid) != self.code_dims:
                        continue

                    for d, v in enumerate(semid):
                        self.dim_unique_values[d].add(int(v))

                    item_key = _row_item_key(date_name, file_idx, row_idx)
                    self.semid_to_items[str(semid)].add(item_key)

        if self.code_dims is None:
            raise RuntimeError("未发现有效 semid。请检查数据目录与字段。")

        if self.expected_code_dims and self.expected_code_dims != self.code_dims:
            print(f"[提示] 期望 code_dims={self.expected_code_dims}，实际={self.code_dims}，以实际为准。")

        print(f"数据行数: {total_rows:,}，semid 维度: {self.code_dims}")
        self._build_vocab()

        mappings = {
            "code_dims": self.code_dims,
            "dim_unique_sizes": [len(s) for s in self.dim_unique_values],
            "scanned_rows": total_rows
        }
        with open(os.path.join(self.output_dir, "mappings.json"), "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)

        semid_to_items_out = {k: sorted(list(v)) for k, v in self.semid_to_items.items()}
        with open(os.path.join(self.output_dir, "semid_to_items.json"), "w", encoding="utf-8") as f:
            json.dump(semid_to_items_out, f, indent=2, ensure_ascii=False)

        print("已保存: mappings.json, semid_to_items.json")

    def _build_vocab(self):
        token_id = 0
        self.special_tokens = {"<PAD>": token_id}; token_id += 1
        self.special_tokens["<BOS>"] = token_id; token_id += 1
        self.special_tokens["<EOS>"] = token_id; token_id += 1

        self.user_bucket_tokens = []
        for _ in range(self.user_bucket_count):
            self.user_bucket_tokens.append(token_id)
            token_id += 1

        self.code_value_to_token = []
        self.token_to_code_value = []
        for d in range(self.code_dims):
            v2t, t2v = {}, {}
            for v in sorted(self.dim_unique_values[d]):
                v2t[v] = token_id
                t2v[token_id] = v
                token_id += 1
            self.code_value_to_token.append(v2t)
            self.token_to_code_value.append(t2v)

        vocab = {
            "special_tokens": self.special_tokens,
            "user_bucket_tokens": self.user_bucket_tokens,
            "code_dims": self.code_dims,
            "dim_value_to_token": [{str(k): v for k, v in d.items()} for d in self.code_value_to_token],
        }
        with open(os.path.join(self.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        print(f"词表已生成，大小约: {token_id:,}")

    # ---------------- 构建用户序列（时间序） ----------------

    def build_user_semid_sequences(self):
        print("【2/5】按时间构建用户序列...")
        user_sequences = defaultdict(list)
        total = 0

        for semid_dir in tqdm(self.semid_data_dirs, desc="生成序列"):
            date_name = os.path.basename(semid_dir.rstrip("/"))
            files = self._scan_parquets(semid_dir)
            for file_idx, fp in enumerate(tqdm(files, desc=date_name, leave=False)):
                try:
                    df = pd.read_parquet(fp)
                except Exception as e:
                    print(f"[跳过] 读取失败: {fp} ({e})")
                    continue
                if "md5_oaid" not in df.columns or "semid" not in df.columns:
                    print(f"[跳过] 缺少必要列(md5_oaid/semid): {fp}")
                    continue

                for row_idx, row in df.iterrows():
                    total += 1
                    uid = row["md5_oaid"]
                    semid_raw = row["semid"]
                    if isinstance(semid_raw, (list, tuple, np.ndarray)):
                        semid = tuple(int(x) for x in semid_raw)
                    else:
                        try:
                            s = str(semid_raw).strip().strip("[]")
                            semid = tuple(int(x) for x in s.split(",") if str(x).strip() != "")
                        except Exception:
                            continue
                    if len(semid) != self.code_dims:
                        continue
                    ts = _row_item_key(date_name, file_idx, row_idx)
                    user_sequences[uid].append({"timestamp": ts, "semid": semid})

        for uid in user_sequences:
            user_sequences[uid].sort(key=lambda x: x["timestamp"])

        print(f"总交互: {total:,}，用户数(未过滤): {len(user_sequences):,}")
        return user_sequences

    # ---------------- token 化与样本构造 ----------------

    def _user_to_bucket_token(self, user_id: str) -> int:
        h = hashlib.md5(str(user_id).encode("utf-8")).hexdigest()
        bucket = int(h, 16) % self.user_bucket_count
        return self.user_bucket_tokens[bucket]

    def _semid_to_tokens(self, semid_tuple):
        toks = []
        for d, v in enumerate(semid_tuple):
            tok = self.code_value_to_token[d].get(int(v))
            if tok is None:
                continue
            toks.append(tok)
        return toks

    def _tokenize_per_item(self, seq_semid):
        per_item_tokens = []
        for it in seq_semid:
            toks = self._semid_to_tokens(it["semid"])
            if len(toks) == self.code_dims:
                per_item_tokens.append(toks)
        return per_item_tokens

    def _samples_from_per_item(self, uid, per_item, index_list, max_seq_tokens):
        user_tok = self._user_to_bucket_token(uid)
        out = []

        for i in index_list:
            if self.max_items is not None and self.max_items > 0:
                left = max(0, i - self.max_items)
            else:
                left = 0
            history_items = per_item[left:i]

            hist_tokens = [user_tok]
            for itoks in history_items:
                hist_tokens.extend(itoks)

            if len(hist_tokens) > max_seq_tokens:
                hist_tokens = hist_tokens[-max_seq_tokens:]

            target_tokens = per_item[i][: self.code_dims]
            out.append({
                "user_id": uid,
                "history_tokens": hist_tokens,
                "target_tokens": target_tokens,
                "history_len": len(hist_tokens),
            })
        return out

    def make_samples_token_level_leave_one_out(self, user_semid_sequences, max_seq_tokens=None):
        print("【3/5】按用户 leave-one-out 生成样本...")
        if max_seq_tokens is None:
            max_seq_tokens = self.max_seq_tokens

        train, val, test = [], [], []
        kept_users = 0

        for uid, seq in tqdm(user_semid_sequences.items(), desc="构样本"):
            per_item = self._tokenize_per_item(seq)
            if len(per_item) < max(self.min_items, 2):
                continue
            kept_users += 1
            L = len(per_item)

            # 训练用目标索引
            if L >= 3:
                train_idx = list(range(1, L - 2))  # [1 .. L-3]
            else:
                train_idx = []

            val_idx = [L - 2] if L >= 2 else []
            test_idx = [L - 1] if L >= 1 else []

            train += self._samples_from_per_item(uid, per_item, train_idx, max_seq_tokens)
            if val_idx:
                val += self._samples_from_per_item(uid, per_item, val_idx, max_seq_tokens)
            if test_idx:
                test += self._samples_from_per_item(uid, per_item, test_idx, max_seq_tokens)

        print(f"保留用户(>= {self.min_items} 个 item): {kept_users:,}")
        print(f"样本统计 | train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
        return {"train": train, "validation": val, "test": test}

    # ---------------- 兼容：保存用户扁平序列 ----------------

    def save_user_sequences_tokens(self, user_semid_sequences):
        print("【4/5】保存用户扁平 token 序列(兼容/排查用)...")
        user_token_sequences = {}
        kept = 0
        for uid, seq in tqdm(user_semid_sequences.items(), desc="扁平化"):
            per_item = self._tokenize_per_item(seq)
            if len(per_item) < max(self.min_items, 1):
                continue
            toks = [self._user_to_bucket_token(uid)]
            for itoks in per_item:
                toks.extend(itoks)
            user_token_sequences[uid] = toks
            kept += 1

        for split_name in ["train", "validation", "test"]:
            fp = os.path.join(self.output_dir, f"{split_name}_sequences_tokens.pkl")
            with open(fp, "wb") as f:
                pickle.dump(user_token_sequences, f)
        print(f"已保存用户扁平序列，用户数: {kept:,}")
        return kept

    # ---------------- 保存样本 ----------------

    def save_samples(self, split_name, samples):
        fp = os.path.join(self.output_dir, f"{split_name}_samples_tokens.pkl")
        with open(fp, "wb") as f:
            pickle.dump(samples, f)
        print(f"[OK] {split_name}_samples_tokens.pkl -> {fp}")


def main():
    parser = argparse.ArgumentParser(description="构建用于阶段二训练的序列样本（semid-only）")
    parser.add_argument("--semid_data_dirs", nargs="+", default=[
        "/mnt/xieshen/data_processed/shop_0814v4/20250706",
        "/mnt/xieshen/data_processed/shop_0814v4/20250707",
        "/mnt/xieshen/data_processed/shop_0814v4/20250708",
        "/mnt/xieshen/data_processed/shop_0814v4/20250709",
        "/mnt/xieshen/data_processed/shop_0814v4/20250710",
        "/mnt/xieshen/data_processed/shop_0814v4/20250711",
        "/mnt/xieshen/data_processed/shop_0814v4/20250712",
    ])
    parser.add_argument("--output_dir", default="./sequence_data")
    parser.add_argument("--min_items", type=int, default=5)
    parser.add_argument("--max_items", type=int, default=10)
    parser.add_argument("--max_seq_tokens", type=int, default=64)
    parser.add_argument("--expected_code_dims", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20250815)
    args = parser.parse_args()

    start_time = time.time()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("构建序列（SEMID 版本）")
    print("=" * 80)

    builder = SequenceBuilder(
        semid_data_dirs=args.semid_data_dirs,
        output_dir=args.output_dir,
        min_items=args.min_items,
        max_items=args.max_items,
        user_bucket_count=2000,
        max_seq_tokens=args.max_seq_tokens,
        expected_code_dims=args.expected_code_dims
    )

    # 1) 读取并建立词表、映射文件
    builder.load_semid_and_build_vocab()

    # 2) 构建用户时间序列
    user_semid_sequences = builder.build_user_semid_sequences()

    # 3) LOO 构造样本并保存
    samples = builder.make_samples_token_level_leave_one_out(user_semid_sequences, builder.max_seq_tokens)
    print("【5/5】写出样本...")
    for split_name in ["train", "validation", "test"]:
        builder.save_samples(split_name, samples[split_name])

    # 4) 兼容保存扁平 token 序列（非训练输入，仅用于排查/统计）
    kept_users_for_seq = builder.save_user_sequences_tokens(user_semid_sequences)

    elapsed = time.time() - start_time

    # 汇总信息
    summary = {
        "semid_data_dirs": args.semid_data_dirs,
        "output_dir": args.output_dir,
        "min_items": args.min_items,
        "max_items": args.max_items,
        "max_seq_tokens": args.max_seq_tokens,
        "code_dims": builder.code_dims,
        "dim_unique_sizes": [len(s) for s in builder.dim_unique_values],
        "users_for_sequences_tokens": kept_users_for_seq,
        "samples": {
            "train": len(samples["train"]),
            "validation": len(samples["validation"]),
            "test": len(samples["test"]),
        },
        "vocab": {
            "special": builder.special_tokens,
            "user_bucket_count": builder.user_bucket_count,
        },
        "time_cost_sec": round(elapsed, 2)
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== 运行完成 ===")
    print(f"样本统计 | train: {summary['samples']['train']:,}  "
          f"val: {summary['samples']['validation']:,}  "
          f"test: {summary['samples']['test']:,}")
    print(f"code 维度: {builder.code_dims}，各维取值规模: {summary['dim_unique_sizes']}")
    print(f"词表与映射: vocab.json, mappings.json, semid_to_items.json")
    print(f"兼容输出: train/validation/test_sequences_tokens.pkl")
    print(f"摘要: run_summary.json")
    print(f"耗时: {summary['time_cost_sec']} 秒")
    print("=" * 80)


if __name__ == "__main__":
    main()