from torch import nn
from torch import Tensor
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        """
        默认用 mean，使数值更平衡。
        reduction: "mean" 或 "sum"
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        diff = (x_hat - x) ** 2
        if self.reduction == "mean":
            return diff.mean(dim=-1)   # 每个样本一个标量
        else:
            return diff.sum(dim=-1)


class CategoricalReconstuctionLoss(nn.Module):
    def __init__(self, n_cat_feats: int, reduction: str = "mean") -> None:
        """
        混合连续特征的 MSE + 类别特征的 BCE
        """
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss(reduction=reduction)
        self.n_cat_feats = n_cat_feats
        self.reduction = reduction

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        # 连续特征
        reconstr = self.reconstruction_loss(
            x_hat[:, :-self.n_cat_feats],
            x[:, :-self.n_cat_feats],
        )

        # 类别特征
        if self.n_cat_feats > 0:
            cat_reconstr = F.binary_cross_entropy_with_logits(
                x_hat[:, -self.n_cat_feats:],
                x[:, -self.n_cat_feats:],
                reduction="none"
            ).mean(dim=-1 if self.reduction == "mean" else 1)
            reconstr = reconstr + cat_reconstr

        return reconstr


class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0, scale: float = 1.0) -> None:
        """
        commitment_weight: encoder 往 codebook 靠拢的权重
        scale: 全局缩放因子，方便调 recon / vq 平衡
        """
        super().__init__()
        self.commitment_weight = commitment_weight
        self.scale = scale

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        # embedding loss: 更新 codebook
        emb_loss = ((query.detach() - value) ** 2).mean(dim=-1)
        # commitment loss: 更新 encoder
        query_loss = ((query - value.detach()) ** 2).mean(dim=-1)

        loss = emb_loss + self.commitment_weight * query_loss
        return self.scale * loss.mean()
