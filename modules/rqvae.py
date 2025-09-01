# rqvae.py

import math
from typing import List, NamedTuple, Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange
from functools import cached_property

from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss, ReconstructionLoss
from modules.normalize import l2norm
from modules.quantize import Quantize, QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin

torch.set_float32_matmul_precision("high")


class RqVaeOutput(NamedTuple):
    embeddings: Tensor      # [n_layers, embed_dim, batch]
    residuals: Tensor       # [n_layers, embed_dim, batch]
    sem_ids: Tensor         # [n_layers, batch]
    quantize_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor
    cosine_sim: Tensor
    rmse: Tensor
    quantization_error: Tensor
    recon_error_p50: Tensor
    recon_error_p90: Tensor
    recon_error_p99: Tensor


class RqVae(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 0,
        # === 统计与日志 ===
        util_log_every: int = 50,   # 每多少步打印一次；<=0 则不打印
        util_decay: float = 0.99,    # EMA 衰减
        util_recent_n: int = 5       # 仅保留最近 N 行
    ) -> None:
        super().__init__()
        self._config = locals()

        # 基本参数
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features

        # 量化层（RQ）
        self.layers = nn.ModuleList(
            [
                Quantize(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    forward_mode=codebook_mode,
                    do_kmeans_init=codebook_kmeans_init,
                    codebook_normalize=(i == 0 and codebook_normalize),
                    sim_vq=codebook_sim_vq,
                    commitment_weight=commitment_weight,
                )
                for i in range(n_layers)
            ]
        )

        # 编码器 / 解码器
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize,
        )
        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[::-1],
            out_dim=input_dim,
            normalize=True,
        )

        # 重构损失
        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features)
            if n_cat_features != 0
            else ReconstructionLoss()
        )

        # === 日志控制 ===
        self.util_log_every = int(util_log_every)
        self.util_decay = float(util_decay)
        self.util_recent_n = int(util_recent_n)
        self._step = 0
        self.util_recent_logs: List[str] = []

        # === 用于 EMA 的统计缓冲区 ===
        # counts: [n_layers, codebook_size]；tokens: [n_layers]
        self.register_buffer(
            "util_counts_ema",
            torch.zeros(n_layers, codebook_size, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "util_token_ema",
            torch.zeros(n_layers, dtype=torch.float32),
            persistent=False,
        )

    @cached_property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    # ---------- 公开 API ----------
    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model"])
        print(f"---Loaded RQVAE Iter {state.get('iter', 'unknown')}---")

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    # ---------- 前向分解 ----------
    def get_semantic_ids(self, x: Tensor, gumbel_t: float = 0.001) -> RqVaeOutput:
        res = self.encode(x)

        quantize_loss = 0.0
        embs, residuals, sem_ids = [], [], []

        for layer in self.layers:
            residuals.append(res)
            q = layer(res, temperature=gumbel_t)  # q.ids: [batch]
            quantize_loss = quantize_loss + q.loss
            emb, idx = q.embeddings, q.ids
            res = res - emb
            sem_ids.append(idx)
            embs.append(emb)

        return RqVaeOutput(
            embeddings=rearrange(torch.stack(embs, dim=0), "h b d -> h d b"),
            residuals=rearrange(torch.stack(residuals, dim=0), "h b d -> h d b"),
            sem_ids=rearrange(torch.stack(sem_ids, dim=0), "h b -> h b"),
            quantize_loss=torch.as_tensor(quantize_loss),
        )

    # ---------- forward ----------
    def forward(self, batch: Any, gumbel_t: float) -> RqVaeComputedLosses:
        x = batch if isinstance(batch, torch.Tensor) else batch.x

        qout = self.get_semantic_ids(x, gumbel_t)
        embs, sem_ids = qout.embeddings, qout.sem_ids  # embs: [L, D, B], sem_ids: [L, B]

        # 解码（将各层量化向量求和当作重构输入）
        x_hat = self.decode(embs.sum(dim=0).T)  # [B, D] -> decode expects [B, D]
        if self.n_cat_feats > 0:
            x_hat = torch.cat(
                [l2norm(x_hat[..., :-self.n_cat_feats]), x_hat[..., -self.n_cat_feats:]],
                dim=-1,
            )
        else:
            x_hat = l2norm(x_hat)

        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = qout.quantize_loss
        loss = (reconstruction_loss + rqvae_loss).mean()

        # 一些可视化指标
        with torch.no_grad():
            embs_norm = embs.norm(dim=1)  # [L, B]
            # 统计样本是否有完全一样的 code 组合
            same = (
                rearrange(sem_ids, "l b -> b l") == rearrange(sem_ids, "l b -> b l").unsqueeze(1)
            ).all(dim=-1)
            # 去上三角（不含对角）
            p_unique_ids = (~torch.triu(same, diagonal=1)).all(dim=1).sum() / sem_ids.shape[1]

            # === 新增重构质量指标 ===
            # 1. 余弦相似度
            cosine_sim = F.cosine_similarity(x, x_hat, dim=-1).mean()
        
            # 2. RMSE
            rmse = torch.sqrt(F.mse_loss(x, x_hat))
        
            # 3. 量化误差 (编码后的向量与量化向量的距离)
            z_encoded = self.encode(x)  # 编码向量
            z_quantized = embs.sum(dim=0).T  # 量化后向量
            quantization_error = torch.sqrt(((z_encoded - z_quantized) ** 2).sum(dim=-1)).mean()
        
            # 4. 重构误差分布
            recon_errors = torch.sqrt(((x - x_hat) ** 2).sum(dim=-1))  # 每个样本的重构误差
            recon_error_p50 = torch.quantile(recon_errors, 0.5)
            recon_error_p90 = torch.quantile(recon_errors, 0.9) 
            recon_error_p99 = torch.quantile(recon_errors, 0.99)

            # 更新并打印困惑度（只保留最近 5 行）
            self._update_and_maybe_log_util(sem_ids)

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstruction_loss.mean(),
            rqvae_loss=rqvae_loss.mean() if torch.is_tensor(rqvae_loss) else torch.as_tensor(rqvae_loss),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            # 新增指标
            cosine_sim=cosine_sim,
            rmse=rmse,
            quantization_error=quantization_error,
            recon_error_p50=recon_error_p50,
            recon_error_p90=recon_error_p90,
            recon_error_p99=recon_error_p99,
        )

    # ---------- 码本使用统计 / 困惑度 ----------
    @torch.no_grad()
    def _update_and_maybe_log_util(self, sem_ids: Tensor):
        """
        sem_ids: LongTensor [n_layers, batch]
        计算每层直方图 -> 概率 -> 熵 -> 困惑度；
        打印当前批次与 EMA 的利用率和 ppl，仅保留最近 util_recent_n 行
        """
        if self.util_log_every <= 0:
            return

        n_layers, batch = sem_ids.shape
        device = self.util_counts_ema.device

        # —— 当前批次直方图（敏感）——
        hists = []
        for k in range(n_layers):
            h = torch.bincount(
                sem_ids[k].reshape(-1),
                minlength=self.codebook_size,
            ).to(device=device, dtype=torch.float32)
            hists.append(h)
        hist = torch.stack(hists, dim=0)  # [L, K]

        # —— EMA 累计（平滑）——
        self.util_counts_ema.mul_(self.util_decay).add_(hist * (1.0 - self.util_decay))
        self.util_token_ema.mul_(self.util_decay).add_(float(batch) * (1.0 - self.util_decay))

        self._step += 1
        if self._step % self.util_log_every != 0:
            return

        eps = 1e-12

        # —— 当前批次利用率 & ppl（敏感）——
        used_batch = (hist > 0).sum(dim=1)  # [L]
        probs_batch = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)  # [L, K]
        H_batch = -(probs_batch * probs_batch.clamp_min(eps).log()).sum(dim=1)
        ppl_batch = torch.exp(H_batch)  # base-e perplexity

        # 额外可读指标（当前批次）
        top1 = probs_batch.max(dim=1).values                                  # 每层 top1 占比
        topk_vals, _ = probs_batch.topk(k=min(5, self.codebook_size), dim=1)  # 前5码占比
        top5 = topk_vals.sum(dim=1)

        # —— EMA 利用率 & ppl（平滑）——
        probs_ema = self.util_counts_ema / self.util_token_ema.clamp_min(eps).unsqueeze(1)
        probs_ema = probs_ema / probs_ema.sum(dim=1, keepdim=True).clamp_min(eps)
        tau = 1e-4  # 阈值，避免 once-used forever
        used_ema = (probs_ema > tau).sum(dim=1)

        H_ema = -(probs_ema * probs_ema.clamp_min(eps).log()).sum(dim=1)
        ppl_ema = torch.exp(H_ema)

        # —— 打印（仅保留最近 N 行）——
        parts = []
        for k in range(n_layers):
            parts.append(
                f"L{k}:{int(used_batch[k])}/{self.codebook_size} "
                f"{(used_batch[k].float()/self.codebook_size*100):.1f}% "
                f"ppl(b)={ppl_batch[k].item():.1f} "
                f"ppl(ema)={ppl_ema[k].item():.1f} "
                f"top1={top1[k].item():.3f} "
                f"top5={top5[k].item():.3f}"
            )
        line = f"step={self._step} | " + " | ".join(parts)

        self.util_recent_logs.append(line)
        if len(self.util_recent_logs) > self.util_recent_n:
            self.util_recent_logs.pop(0)

        print("\n".join(self.util_recent_logs), flush=True)