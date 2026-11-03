"""Квантование по теории шага с пропуском нуля (QAT, STE).

Сетка (теория, разделы 1-4):
    шаг      = n / 2^p        (двоичная дробь -> точное представление)
    значения = step * k,  k in {-2^p, ..., -1, 1, ..., 2^p}   (ноль отсутствует)
    всего    = 2^(p+1) значений  -> ровно p+1 бит, без "мёртвого" кода.

Реальные веса не ограничены [-n, n], поэтому применяется масштаб (теория, разд. 11):
    w = s * step * k,   s - масштаб блока (absmax), fp32/float16 на блок.

Два режима масштаба:
    global    - s = 1, клип в [-n, n] (как в разделе 4 плана);
    per_block - s = absmax блока (32 веса), шаг сетки масштабируется на блок.

Кодирование c <-> k (теория, разд. 4):
    c = k + 2^p      при k < 0
    c = k + 2^p - 1  при k > 0
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Кодирование "p+1 бит" (теория, раздел 4)
# ----------------------------------------------------------------------------
def encode_k(k: torch.Tensor, p: int) -> torch.Tensor:
    """k in {-2^p..-1, 1..2^p} -> code in [0, 2^(p+1)-1]."""
    shift = 2 ** p
    k = k.to(torch.int64)
    return torch.where(k < 0, k + shift, k + shift - 1)


def decode_c(c: torch.Tensor, p: int) -> torch.Tensor:
    """code in [0, 2^(p+1)-1] -> k in {-2^p..-1, 1..2^p}."""
    shift = 2 ** p
    c = c.to(torch.int64)
    return torch.where(c <= shift - 1, c - shift, c - shift + 1)


def encode_k_with_zero(k: torch.Tensor, p: int) -> torch.Tensor:
    """Вариант Б: сетка с нулём, k in [-2^p, 2^p] -> код 0..2^(p+1) (33 значения).

    Это не финальное кодирование варианта Б (в теории, разд. 7, ноль кодируется
    1 битом, ненулевое значение - p+2 битами); здесь код нужен только для
    сохранения весов без потерь.
    """
    return (k.to(torch.int64) + 2 ** p)


def decode_c_with_zero(c: torch.Tensor, p: int) -> torch.Tensor:
    return c.to(torch.int64) - 2 ** p


def average_bits_with_zero(p: int, zero_frac: float) -> float:
    """Средняя длина кода варианта Б с вырезанием нуля (теория, разд. 7)."""
    return zero_frac * 1.0 + (1.0 - zero_frac) * (p + 2)


def bits_per_weight(p: int, skip_zero: bool) -> float:
    """Число бит на вес (без учёта масштаба блока)."""
    if skip_zero:
        return float(p + 1)          # 2^(p+1) значений, все коды заняты
    return math.log2(2 ** (p + 1) + 1)  # сетка с нулём: +1 значение


# ----------------------------------------------------------------------------
# Квантователь
# ----------------------------------------------------------------------------
class StepQuantizer(nn.Module):
    """Равномерная сетка k * step с пропуском нуля (по умолчанию).

    forward(w) возвращает квантованные веса с прямым проходом градиента (STE):
        q = w + (q_detach - w).detach()
    """

    def __init__(
        self,
        n: float = 1.0,
        p: int = 4,
        skip_zero: bool = True,
        scale_mode: str = "per_block",
        block_size: int = 32,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if scale_mode not in ("global", "per_block"):
            raise ValueError(f"scale_mode должен быть global|per_block, получено {scale_mode!r}")
        if block_size <= 0:
            raise ValueError("block_size должен быть > 0")
        self.n = float(n)
        self.p = int(p)
        self.skip_zero = bool(skip_zero)
        self.scale_mode = scale_mode
        self.block_size = int(block_size)
        self.eps = float(eps)
        self.step = self.n / (2.0 ** self.p)
        self.max_k = 2 ** self.p

    # -- вспомогательное -----------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n={self.n}, p={self.p}, step={self.step:g}, max_k={self.max_k}, "
            f"skip_zero={self.skip_zero}, scale_mode={self.scale_mode}, "
            f"block_size={self.block_size}"
        )

    @torch.no_grad()
    def compute_scale(self, w: torch.Tensor) -> torch.Tensor:
        """Масштаб, приводящий блок весов в диапазон сетки [-n, n]."""
        if self.scale_mode == "global":
            return torch.ones((), dtype=w.dtype, device=w.device)
        flat = w.reshape(-1)
        bs = self.block_size
        pad = (-flat.numel()) % bs
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blocks = flat.view(-1, bs)
        scale = blocks.abs().amax(dim=1).clamp_min(self.eps) / self.n
        return scale.repeat_interleave(bs)[: w.numel()].view_as(w)

    def _k_from_u(self, u: torch.Tensor) -> torch.Tensor:
        """u - нормированные веса (диапазон [-n, n]) -> целые k на сетке."""
        k = torch.round(torch.clamp(u / self.step, -self.max_k, self.max_k))
        if self.skip_zero:
            # ноль отсутствует в сетке: округление в 0 -> ближайший +-1 по знаку
            sign = torch.where(u < 0, -torch.ones_like(k), torch.ones_like(k))
            k = torch.where(k == 0, sign, k)
        return k

    # -- прямой проход -------------------------------------------------------
    def quantize_details(self, w: torch.Tensor):
        """Возвращает (q, k, u, scale) для диагностики."""
        scale = self.compute_scale(w)
        u = w / scale
        k = self._k_from_u(u)
        q = k * self.step * scale
        return q, k, u, scale

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        q, _, _, _ = self.quantize_details(w)
        return w + (q - w).detach()  # STE


class QuantLinear(nn.Module):
    """nn.Linear, у которого вес проходит через квантователь; bias остаётся fp32."""

    is_quant = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantizer: StepQuantizer,
        bias: bool = True,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features) if weight is None else weight.clone()
        )
        if weight is None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantizer(self.weight), self.bias)

    @torch.no_grad()
    def quantized_weight(self) -> torch.Tensor:
        return self.quantizer(self.weight)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"


# ----------------------------------------------------------------------------
# Диагностика (план, раздел 7)
# ----------------------------------------------------------------------------
@dataclass
class QuantStats:
    weights: int
    on_grid_frac: float
    zero_frac: float
    satur_frac: float          # доля весов на границе +-n (клип)
    max_abs_norm: float        # max |u| / n
    mean_abs_k: float
    rel_err: float             # sum|q-w| / sum|w| — относительная ошибка квантования
    k_hist: list[int]          # индекс i -> k = i - 2^p  (2^(p+1)+1 корзина)
    grad_norm: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def collect_stats(quant_linears: Iterable[QuantLinear], p: int, skip_zero: bool) -> QuantStats:
    total = 0
    on_grid = 0
    zeros = 0
    satur = 0
    max_abs_norm = 0.0
    abs_k_sum = 0.0
    err_sum = 0.0
    w_sum = 0.0
    n_bins = 2 * 2 ** p + 1
    hist = torch.zeros(n_bins, dtype=torch.long)
    shift = 2 ** p

    for mod in quant_linears:
        w = mod.weight.detach()
        q, k, u, _ = mod.quantizer.quantize_details(w)
        total += w.numel()

        # 1) 100% весов должны лежать точно на сетке:
        #    q / step / scale - целое k  ->  сравниваем q с k*step*scale
        step, scale = mod.quantizer.step, mod.quantizer.compute_scale(w)
        q_ref = torch.round(q / scale / step) * step * scale
        on_grid += torch.isclose(q, q_ref, rtol=1e-6, atol=1e-8).sum().item()

        zeros += (k == 0).sum().item()
        satur += (k.abs() == 2 ** p).sum().item()
        max_abs_norm = max(max_abs_norm, (u.abs().max() / mod.quantizer.n).item())
        abs_k_sum += k.abs().sum().item()
        err_sum += (q - w).abs().sum().item()
        w_sum += w.abs().sum().item()
        hist += torch.bincount((k + shift).reshape(-1).to(torch.int64), minlength=n_bins)[:n_bins].cpu()

    return QuantStats(
        weights=total,
        on_grid_frac=on_grid / max(total, 1),
        zero_frac=zeros / max(total, 1),
        satur_frac=satur / max(total, 1),
        max_abs_norm=max_abs_norm,
        mean_abs_k=abs_k_sum / max(total, 1),
        rel_err=err_sum / max(w_sum, 1e-12),
        k_hist=[int(v) for v in hist.tolist()],
    )


@torch.no_grad()
def grad_health(quant_linears: Iterable[QuantLinear]) -> float:
    """Суммарная норма градиента квантуемых весов (проверка STE: должно быть > 0)."""
    sq = 0.0
    for mod in quant_linears:
        if mod.weight.grad is not None:
            sq += mod.weight.grad.detach().float().pow(2).sum().item()
    return math.sqrt(sq)
