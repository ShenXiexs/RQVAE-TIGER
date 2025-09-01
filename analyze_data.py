# analyze_data.py
"""
步骤1: 数据分析与统计（仅使用“已包含 semid 的数据集”）
功能:
  - 读取带 semid 的 parquet（这些是原始数据基础上直接多了两列）
  - 统计用户数、序列长度分布、semid 维度与取值分布
  - 构建用户的 semid 序列（按时间顺序）
  - 输出摘要 JSON、用户统计 CSV 和可视化 PNG

输入:
  --semid_data_dirs  形如 /mnt/xieshen/data_processed/shop_0814v4/20250706 ... /20250712
  每个目录里是一批 *_parsed.parquet（或 *_semid.parquet / .parquet）文件，且每行至少包含:
    - md5_oaid: 用户ID
    - semid:    语义ID（list/tuple/ndarray），例如 3 维或 4 维

输出:
  ./analysis_results/analysis_summary.json
  ./analysis_results/user_statistics.csv
  ./analysis_results/sequence_length_analysis.png
"""

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------- 核心：只读“带 semid 的数据集” ----------

def scan_semid_parquets(semid_dir: str):
    """返回目录下所有候选 parquet 文件（带有 semid 的数据）。"""
    if not os.path.exists(semid_dir):
        return []

    # 兼容三种常见命名；若没有匹配，退化为所有 .parquet
    cands = sorted([f for f in os.listdir(semid_dir) if f.endswith("_parsed.parquet")])
    if not cands:
        cands = sorted([f for f in os.listdir(semid_dir) if f.endswith("_semid.parquet")])
    if not cands:
        cands = sorted([f for f in os.listdir(semid_dir) if f.endswith(".parquet")])
    return [os.path.join(semid_dir, f) for f in cands]


def analyze_semid_only(semid_dirs):
    """
    只读取带 semid 的 parquet，产出：
      - user_interactions: {user_id: [ {timestamp, date, file_idx, row_idx, semid_tuple}, ... ]}
      - daily_stats: {date: {...}}
      - semid_dim, dim_value_counts
    """
    print("Analyzing data (semid-only mode)...")

    user_interactions = defaultdict(list)
    daily_stats = {}
    total_rows = 0
    total_users_set = set()

    # 用于统计 semid 的维度&取值分布
    semid_dim = None
    dim_value_counts = []  # list[defaultdict(int)]  长度=semid_dim

    for semid_dir in tqdm(semid_dirs, desc="Processing semid date directories"):
        date_name = os.path.basename(semid_dir.rstrip("/"))
        files = scan_semid_parquets(semid_dir)

        daily_users = set()
        daily_cnt = 0

        for file_idx, fp in enumerate(tqdm(files, desc=f"{date_name}", leave=False)):
            try:
                df = pd.read_parquet(fp)
            except Exception as e:
                print(f"Failed to read parquet {fp}: {e}")
                continue

            # 基础列存在性校验
            if "md5_oaid" not in df.columns or "semid" not in df.columns:
                # 直接跳过不合格的文件
                print(f"Skip file without required columns (md5_oaid/semid): {fp}")
                continue

            # 逐行迭代，构建序列元素
            for row_idx, row in df.iterrows():
                user_id = row["md5_oaid"]

                # 统一把 semid 转为 tuple；遇到非 list/ndarray 的情况尝试容错
                raw_semid = row["semid"]
                if isinstance(raw_semid, (list, tuple, np.ndarray)):
                    semid_tuple = tuple(int(x) for x in raw_semid)
                else:
                    # 可能是字符串形式，如 "1,2,3" 或 "[1,2,3]"
                    try:
                        if isinstance(raw_semid, str):
                            s = raw_semid.strip().strip("[]")
                            semid_tuple = tuple(int(x) for x in s.split(",") if str(x).strip() != "")
                        else:
                            continue
                    except Exception:
                        continue

                # 初始化 semid 维度统计器
                if semid_dim is None:
                    semid_dim = len(semid_tuple)
                    dim_value_counts = [defaultdict(int) for _ in range(semid_dim)]

                # 累计维度取值分布
                if len(semid_tuple) == semid_dim:
                    for d, v in enumerate(semid_tuple):
                        dim_value_counts[d][v] += 1

                # 构造严格单调的“时间戳”键（日期_文件序号_行号）
                timestamp = f"{date_name}_{file_idx:03d}_{int(row_idx):06d}"

                user_interactions[user_id].append({
                    "timestamp": timestamp,
                    "date": date_name,
                    "file_idx": file_idx,
                    "row_idx": int(row_idx),
                    "semid": semid_tuple
                })

                daily_cnt += 1
                total_rows += 1
                daily_users.add(user_id)
                total_users_set.add(user_id)

        daily_stats[date_name] = {
            "interactions": daily_cnt,
            "unique_users": len(daily_users),
            "files_processed": len(files),
        }

    # 排序为严格时间序
    for uid in user_interactions:
        user_interactions[uid].sort(key=lambda x: x["timestamp"])

    print("\nData summary (semid-only):")
    print(f"   Total interactions (rows): {total_rows:,}")
    print(f"   Total users: {len(total_users_set):,}")

    return user_interactions, daily_stats, semid_dim, dim_value_counts


# ---------- 派生统计与可视化 ----------

def analyze_sequence_lengths(user_interactions):
    seq_lens = [len(v) for v in user_interactions.values()] or [0]
    stats = {
        "min_length": int(np.min(seq_lens)) if seq_lens else 0,
        "max_length": int(np.max(seq_lens)) if seq_lens else 0,
        "mean_length": float(np.mean(seq_lens)) if seq_lens else 0.0,
        "median_length": float(np.median(seq_lens)) if seq_lens else 0.0,
        "std_length": float(np.std(seq_lens)) if seq_lens else 0.0,
    }

    # 区间统计（可按需要调整区间）
    bins = [1, 2, 5, 10, 20, 50, 100, float("inf")]
    dist = {}
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        cnt = sum(1 for L in seq_lens if a <= L < b)
        dist[f"{a}-{int(b) if b!=float('inf') else 'inf'}"] = cnt

    print("\nSequence length stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    print("\nSequence length distribution:")
    total_users = len(seq_lens)
    for rng, cnt in dist.items():
        pct = (cnt / total_users * 100) if total_users else 0.0
        print(f"   {rng}: {cnt:,} users ({pct:.1f}%)")

    return stats, seq_lens, dist


def generate_visualizations(seq_lens, dist, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 直方图 + 饼图（简单 matplotlib）
    plt.figure(figsize=(12, 6))

    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(seq_lens, bins=50, alpha=0.8, edgecolor="black")
    plt.xlabel("Sequence Length")
    plt.ylabel("Number of Users")
    plt.title("User Sequence Length Distribution")
    plt.yscale("log")

    # 饼图
    plt.subplot(1, 2, 2)
    labels = list(dist.keys())
    sizes = list(dist.values())
    if sum(sizes) == 0:
        sizes = [1] * len(labels)  # 避免空数据报错
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Sequence Length Range Distribution")

    plt.tight_layout()
    out_png = os.path.join(output_dir, "sequence_length_analysis.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_png}")


def save_analysis_results(user_interactions, stats, dist, semid_dim, dim_value_counts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 概览 JSON
    summary = {
        "total_users": len(user_interactions),
        "total_interactions": int(sum(len(v) for v in user_interactions.values())),
        "sequence_stats": stats,
        "length_distribution": dist,
        "semid_dim": int(semid_dim) if semid_dim is not None else None,
        "dim_top5": {}
    }

    if semid_dim is not None:
        for d in range(semid_dim):
            # 取 top5
            counts = dim_value_counts[d]
            top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            summary["dim_top5"][f"dim_{d}"] = [{"value": int(v), "count": int(c)} for v, c in top5]

    with open(os.path.join(output_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("   Saved: analysis_summary.json")

    # 每个用户的序列长度 + 涉及天数
    rows = []
    for uid, seq in user_interactions.items():
        rows.append({
            "user_id": uid,
            "sequence_length": len(seq),
            "date_span": len(set(x["date"] for x in seq)),
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "user_statistics.csv"), index=False)
    print("   Saved: user_statistics.csv")


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser(description="Analyze semid-augmented interaction data")
    parser.add_argument(
        "--semid_data_dirs", nargs="+",
        default=[
            "/mnt/xieshen/data_processed/shop_0814v4/20250706",
            "/mnt/xieshen/data_processed/shop_0814v4/20250707",
            "/mnt/xieshen/data_processed/shop_0814v4/20250708",
            "/mnt/xieshen/data_processed/shop_0814v4/20250709",
            "/mnt/xieshen/data_processed/shop_0814v4/20250710",
            "/mnt/xieshen/data_processed/shop_0814v4/20250711",
            "/mnt/xieshen/data_processed/shop_0814v4/20250712",
        ],
        help="Directories that contain parquet files with BOTH original fields and the `semid` column."
    )
    parser.add_argument("--output_dir", default="./analysis_results", help="Where to store analysis outputs")
    args = parser.parse_args()

    print("=" * 80)
    print("Sequential Recommendation Data Analysis (SEMID-ONLY)")
    print("=" * 80)

    # 只从 semid 数据中读取并构建用户序列
    user_interactions, daily_stats, semid_dim, dim_value_counts = analyze_semid_only(args.semid_data_dirs)

    # 序列长度分析
    stats, seq_lens, dist = analyze_sequence_lengths(user_interactions)

    # 可视化
    generate_visualizations(seq_lens, dist, args.output_dir)

    # 保存结果
    save_analysis_results(user_interactions, stats, dist, semid_dim, dim_value_counts, args.output_dir)

    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()