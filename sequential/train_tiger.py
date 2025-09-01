#!/usr/bin/env python3
# train_tiger.py
"""
步骤5: TIGER模型训练（自回归 seq2seq）
输入: 处理好的序列数据 (encoder_input_ids / encoder_attention_mask / decoder_input_ids / labels)
输出: 训练好的模型（best_model.pt 等）与日志
"""

import os
import json
import time
import math
import random
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from dataset_loader import create_data_loaders
from tiger_model import TIGERSeq2SeqRecommender

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False


def set_seed(seed: int =20250815):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时
    
    # 设置CUDNN（会略微降低速度但确保可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置Python哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"所有随机种子已设置为: {seed}")


class TIGERTrainer:
    """TIGER 自回归 seq2seq 训练器（token-level CE）"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.98),
            eps=1e-9
        )

        if config["scheduler"] == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=config["patience"]
                # 移除 verbose=True 以避免警告
            )
        elif config["scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config["num_epochs"], eta_min=config["learning_rate"] * 0.01
            )
        else:
            self.scheduler = None

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses, self.val_losses = [], []
        
        # 添加早停计数器
        self.no_improve_epochs = 0

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = bool(config.get("use_wandb", False) and WANDB_OK)
        if config.get("use_wandb", False) and not WANDB_OK:
            print("[警告] wandb 未安装，已自动关闭。")
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "tiger-recommendation"),
                name=config.get("run_name", f"tiger-seq2seq-{int(time.time())}"),
                config=config
            )

    # ---------------- Core loss ----------------
    def _compute_loss(self, batch) -> torch.Tensor:
        """
        token-level CrossEntropy:
        - encoder_input_ids [B, L_enc], encoder_attention_mask [B, L_enc] (bool, 1=keep)
        - decoder_input_ids [B, L_dec]
        - labels [B, L_dec] (pad位置为 -100)
        """
        enc_ids = batch["encoder_input_ids"].to(self.device)
        enc_mask = batch["encoder_attention_mask"].to(self.device)
        dec_inp = batch["decoder_input_ids"].to(self.device)
        labels  = batch["labels"].to(self.device)

        out = self.model(
            encoder_input_ids=enc_ids,
            encoder_attention_mask=enc_mask,
            decoder_input_ids=dec_inp
        )  # {"logits": [B, L_dec, V]}
        logits = out["logits"]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="mean"
        )
        return loss

    # ---------------- Train / Val ----------------
    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"训练轮次 {self.current_epoch+1}", leave=False)

        for step, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()

            if self.config.get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])

            self.optimizer.step()
            total += loss.item()
            pbar.set_postfix(损失=f"{total/(step+1):.4f}")

            if self.use_wandb and (step % 50 == 0):
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "epoch": self.current_epoch
                })
        return total / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("inf")
        self.model.eval()
        total = 0.0
        for batch in tqdm(self.val_loader, desc="验证中", leave=False):
            loss = self._compute_loss(batch)
            total += loss.item()
        return total / max(len(self.val_loader), 1)

    # ---------------- Loop ----------------
    def train(self):
        print(f"开始训练，共 {self.config['num_epochs']} 轮，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"早停patience: {self.config.get('early_stop_patience', 15)} 轮")
        print(f"学习率调度patience: {self.config.get('patience', 5)} 轮")

        for epoch in range(self.current_epoch, self.config["num_epochs"]):
            self.current_epoch = epoch

            tr = self.train_epoch()
            self.train_losses.append(tr)

            va = self.validate()
            self.val_losses.append(va)

            # 学习率调度
            old_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(va)
                else:
                    self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            
            # 如果学习率发生变化，打印信息
            if old_lr != new_lr:
                print(f"  学习率调整: {old_lr:.2e} -> {new_lr:.2e}")

            print(f"轮次 {epoch+1}/{self.config['num_epochs']}  训练损失: {tr:.4f}  验证损失: {va:.4f}")

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": tr,
                    "val/loss": va,
                    "lr": self.optimizer.param_groups[0]["lr"]
                })

            # 检查是否是新的最佳模型
            if va < self.best_val_loss:
                self.best_val_loss = va
                self.no_improve_epochs = 0  # 重置计数器
                self._save_ckpt("best_model.pt")
                print(f"  ✓ 保存了新的最佳模型 (验证损失: {va:.4f})")
            else:
                self.no_improve_epochs += 1
                print(f"  未改善 (已连续 {self.no_improve_epochs} 轮，最佳: {self.best_val_loss:.4f})")

            # 定期保存检查点
            if (epoch + 1) % self.config["save_every"] == 0:
                self._save_ckpt(f"checkpoint_epoch_{epoch+1}.pt")
                print(f"  定期检查点已保存")

            # 早停检查
            if self.no_improve_epochs >= self.config.get("early_stop_patience", 15):
                print(f"\n连续 {self.no_improve_epochs} 轮没有改善，触发早停机制。")
                print(f"最佳验证损失: {self.best_val_loss:.4f}")
                break

        print("\n训练完成！")
        self._save_ckpt("last_model.pt")
        
        # 打印训练总结
        print("\n" + "="*60)
        print("训练总结")
        print("="*60)
        print(f"总训练轮次: {self.current_epoch + 1}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"最终训练损失: {self.train_losses[-1]:.4f}")
        print(f"最终验证损失: {self.val_losses[-1]:.4f}")

    def _save_ckpt(self, filename: str):
        ckpt = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "no_improve_epochs": self.no_improve_epochs,  # 保存早停计数
            "seed": self.config.get("seed",20250815)  # 保存种子信息
        }
        if self.scheduler:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        path = str(self.output_dir / filename)
        torch.save(ckpt, path)

    def load_ckpt(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.train_losses = ckpt.get("train_losses", [])
        self.val_losses = ckpt.get("val_losses", [])
        self.no_improve_epochs = ckpt.get("no_improve_epochs", 0)  # 恢复早停计数
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"从 {path} 恢复训练")
        print(f"  当前轮次: {self.current_epoch + 1}")
        print(f"  最佳验证损失: {self.best_val_loss:.4f}")
        print(f"  连续未改善轮次: {self.no_improve_epochs}")
        
        # 恢复种子（如果存在）
        if "seed" in ckpt:
            print(f"  使用种子: {ckpt['seed']}")
            set_seed(ckpt['seed'])


# ================= CLI =================

def main():
    parser = argparse.ArgumentParser(description="训练 TIGER seq2seq 模型（token级别）")
    # 数据与输出
    parser.add_argument("--data_dir", default="./sequence_data", help="处理后的数据目录")
    parser.add_argument("--output_dir", default="./tiger_models", help="模型保存目录")
    # 模型
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--max_seq_tokens", type=int, default=64,
                        help="编码器侧扁平token长度上限（与数据生成一致）")
    # 训练
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="plateau")
    parser.add_argument("--patience", type=int, default=5, help="学习率调度器patience")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=15, help="早停patience")
    parser.add_argument("--num_workers", type=int, default=4)
    # 随机种子
    parser.add_argument("--seed", type=int, default=20250815, help="随机种子")
    # 其他
    parser.add_argument("--resume_from", default=None, help="从检查点恢复训练")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="tiger-recommendation")
    parser.add_argument("--run_name", default=None)

    args = parser.parse_args()

    # ========== 设置随机种子 ==========
    set_seed(args.seed)
    
    print("="*80)
    print("TIGER Seq2Seq 模型训练（Token级别）")
    print(f"随机种子: {args.seed}")
    print("="*80)

    # DataLoaders（注意：新版 loader 接口为 max_seq_tokens）
    print("创建数据加载器...")
    train_loader, val_loader, test_loader, meta = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_tokens=args.max_seq_tokens,
        num_workers=args.num_workers,
        seed=args.seed  # 传递种子给数据加载器
    )
    if train_loader is None:
        print("错误：创建数据加载器失败。")
        return

    vocab_size = meta["vocab_size"]
    pad_token_id = meta["pad_token"]

    print(f"词表大小: {vocab_size:,}")
    print(f"训练批次数: {len(train_loader):,}")
    if val_loader:  print(f"验证批次数: {len(val_loader):,}")
    if test_loader: print(f"测试批次数: {len(test_loader):,}")

    # Model
    print("构建模型...")
    model = TIGERSeq2SeqRecommender(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_tokens,
        pad_token_id=pad_token_id,
        tie_weights=True
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # Config for saving
    config = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "max_seq_tokens": args.max_seq_tokens,
        "scheduler": args.scheduler,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
        "save_every": args.save_every,
        "early_stop_patience": args.early_stop_patience,
        "output_dir": args.output_dir,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "run_name": args.run_name or f"tiger_seq2seq_{int(time.time())}",
        "data_dir": args.data_dir,
        "vocab_size": vocab_size,
        "pad_token": pad_token_id,
        "seed": args.seed  # 保存种子到配置
    }

    trainer = TIGERTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )

    if args.resume_from:
        trainer.load_ckpt(args.resume_from)

    trainer.train()

    print("\n" + "="*80)
    print("训练完成！")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"模型保存在: {args.output_dir}")
    print(f"使用的随机种子: {args.seed}")
    print("="*80)


if __name__ == "__main__":
    main()
