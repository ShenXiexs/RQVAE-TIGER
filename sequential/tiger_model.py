# tiger_model.py
"""
步骤4: TIGER推荐模型（对齐"逐 code 自回归"管线）
- TIGERRecommender: 兼容旧的 src/tgt 用法（保留）
- TIGERSeq2SeqRecommender: 新版，自回归 Encoder-Decoder，forward(...) 返回 {"logits": [B, L_dec, V]}
"""

import math
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- Positional Encoding --------------------

class PositionalEncoding(nn.Module):
    """正弦位置编码（batch_first 友好）"""
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [L,D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        x = x + self.pe[:L].unsqueeze(0)  # [1,L,D]
        return self.dropout(x)


# -------------------- 旧版：保留以兼容（可不再使用） --------------------

class TIGERRecommender(nn.Module):
    """
    旧版 Encoder-Decoder（一次只预测一个token），接口沿用原实现。
    现有工程以新版 TIGERSeq2SeqRecommender 为主。
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.)

    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        # [T,T], 上三角为 -inf
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        # src/tgt: [B, L]
        src_emb = self.pos_encoding(self.token_embedding(src) * math.sqrt(self.d_model))
        if tgt is not None:
            tgt_emb = self.pos_encoding(self.token_embedding(tgt) * math.sqrt(self.d_model))
            tgt_mask = self._causal_mask(tgt_emb.size(1), src.device)
            out = self.transformer(
                src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        else:
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
            out = memory[:, -1:, :]  # [B,1,D]
        logits = self.output_projection(out)  # [B, L_tgt, V]
        return {"logits": logits}


# -------------------- 新版：自回归 Encoder-Decoder（主力） --------------------

class TIGERSeq2SeqRecommender(nn.Module):
    """
    自回归序列到序列模型：
    forward(
        encoder_input_ids: [B, L_enc],
        encoder_attention_mask: [B, L_enc] (bool, 1=keep, 0=pad),
        decoder_input_ids: [B, L_dec]
    ) -> {"logits": [B, L_dec, V]}
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
        pad_token_id: int = 0,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # 共享 embedding（encoder/decoder）
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout=dropout)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # 输出投影
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight  # 权重共享

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "weight" in name and "embedding" not in name:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        # [T,T], 上三角为 -inf
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        - encoder_attention_mask: [B, L_enc]，True=有效，False=padding
        - decoder_input_ids: 训练时必须提供（= [BOS] + target_tokens）
        - decoder_attention_mask: 可选（若不提供将依据 PAD 自动生成）
        """
        device = encoder_input_ids.device

        # ----- Encoder -----
        enc_emb = self.pos_encoding(self.token_embedding(encoder_input_ids) * math.sqrt(self.d_model))
        src_key_padding_mask = None
        if encoder_attention_mask is not None:
            # Transformer 期望 True=需要mask（即padding），我们的mask True=keep，所以取反
            src_key_padding_mask = ~encoder_attention_mask  # [B, L_enc] bool

        memory = self.encoder(enc_emb, src_key_padding_mask=src_key_padding_mask)  # [B, L_enc, D]

        # 推理（仅encoder）场景：返回最后一步的 logits（很少在本项目用）
        if decoder_input_ids is None:
            last = memory[:, -1:, :]
            logits = self.output_projection(last)
            return {"logits": logits}

        # ----- Decoder -----
        dec_emb = self.pos_encoding(self.token_embedding(decoder_input_ids) * math.sqrt(self.d_model))
        # 自回因果 mask
        tgt_len = dec_emb.size(1)
        tgt_mask = self._causal_mask(tgt_len, device)

        # decoder padding mask：True=padding
        tgt_key_padding_mask = None
        if decoder_attention_mask is not None:
            tgt_key_padding_mask = ~decoder_attention_mask
        else:
            tgt_key_padding_mask = decoder_input_ids.eq(self.pad_token_id)

        out = self.decoder(
            tgt=dec_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )  # [B, L_dec, D]

        logits = self.output_projection(out)  # [B, L_dec, V]
        return {"logits": logits}

    # 可选：单步下一 token 的分布（供自定义采样/调试）
    @torch.no_grad()
    def next_token_logits(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        out = self.forward(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids
        )["logits"]
        return out[:, -1, :]  # [B, V]


# -------------------- 简单自测 --------------------

def _self_test():
    B, L_enc, L_dec, V = 2, 16, 5, 1000
    print("测试 TIGERSeq2SeqRecommender 模型...")
    model = TIGERSeq2SeqRecommender(vocab_size=V, d_model=256, nhead=8, num_layers=4, max_seq_length=256, pad_token_id=0)
    enc = torch.randint(1, V, (B, L_enc))
    enc_mask = torch.ones(B, L_enc, dtype=torch.bool)
    dec_inp = torch.randint(1, V, (B, L_dec))
    # 人为制造 padding
    dec_inp[0, -1] = 0
    out = model(encoder_input_ids=enc, encoder_attention_mask=enc_mask, decoder_input_ids=dec_inp)
    assert out["logits"].shape == (B, L_dec, V)
    print(f"✓ TIGERSeq2SeqRecommender 测试通过！输出形状: {out['logits'].shape}")
    print(f"  批次大小: {B}")
    print(f"  编码器序列长度: {L_enc}")
    print(f"  解码器序列长度: {L_dec}")
    print(f"  词表大小: {V}")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    print("="*60)
    print("TIGER 模型自测")
    print("="*60)
    _self_test()
    print("="*60)
    print("所有测试通过！")
