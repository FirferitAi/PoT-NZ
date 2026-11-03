"""Крошечный decoder-only Transformer с опциональным QAT на линейных слоях.

Правило замены (план, раздел 11): QuantLinear стоит ТОЛЬКО в attention/FFN;
embedding и lm_head остаются fp32.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import QuantLinear, StepQuantizer


@dataclass
class ModelConfig:
    vocab_size: int = 65
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    context: int = 128
    dropout: float = 0.1
    tie_embeddings: bool = False
    quantize_embedding: bool = False   # квантовать эмбеддинг и lm_head
    quant: dict | None = None  # None -> fp32 baseline


class QuantEmbedding(nn.Module):
    """nn.Embedding, у которого вес проходит через квантователь (теория шага)."""

    is_quant = True

    def __init__(self, num_embeddings: int, embedding_dim: int, quantizer: StepQuantizer) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.quantizer = quantizer

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.quantizer(self.weight))

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class QuantHead(nn.Module):
    """lm_head, делящий квантованный вес с эмбеддингом (tie_embeddings)."""

    def __init__(self, emb: QuantEmbedding) -> None:
        super().__init__()
        self.weight = emb.weight      # тот же Parameter
        self.quantizer = emb.quantizer
        self.in_features = emb.embedding_dim
        self.out_features = emb.num_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantizer(self.weight))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, linear_cls) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = linear_cls(cfg.d_model, 3 * cfg.d_model)
        self.proj = linear_cls(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, linear_cls) -> None:
        super().__init__()
        self.fc1 = linear_cls(cfg.d_model, cfg.d_ff)
        self.fc2 = linear_cls(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, linear_cls) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, linear_cls)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg, linear_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.quantizer: StepQuantizer | None = None
        if cfg.quant is not None:
            self.quantizer = StepQuantizer(**cfg.quant)

        def linear_cls(in_f: int, out_f: int):
            if self.quantizer is None:
                return nn.Linear(in_f, out_f)
            return QuantLinear(in_f, out_f, self.quantizer)

        if cfg.quantize_embedding and self.quantizer is not None:
            self.tok_emb = QuantEmbedding(cfg.vocab_size, cfg.d_model, self.quantizer)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        self.pos_emb = nn.Embedding(cfg.context, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, linear_cls) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        if cfg.quantize_embedding and cfg.tie_embeddings and self.quantizer is not None:
            self.lm_head = QuantHead(self.tok_emb)
        else:
            self.lm_head = linear_cls(cfg.d_model, cfg.vocab_size) if                 (cfg.quantize_embedding and self.quantizer is not None) else                 nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.tok_emb.weight  # общий тензор
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (QuantLinear, QuantEmbedding, QuantHead)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        assert t <= self.cfg.context, f"seq len {t} > context {self.cfg.context}"
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )
        return logits, loss

    # -- удобства -----------------------------------------------------------
    def quant_linears(self):
        """Все модули с собственным квантованным весом (attention/FFN и эмбеддинг)."""
        return [m for m in self.modules()
                if isinstance(m, (QuantLinear, QuantEmbedding))]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_quant_params(self) -> int:
        return sum(m.weight.numel() for m in self.quant_linears())

    @torch.no_grad()
    def quantize_all_(self) -> None:
        """Жёстко записать квантованные веса в параметры (для сохранения/оценки)."""
        for m in self.quant_linears():
            m.weight.copy_(m.quantizer(m.weight))
