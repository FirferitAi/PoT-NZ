# The PoT-NZ method: power-of-two quantization without zero

This document states the method, its algebra, and the thresholds that follow from the
formats. Empirical confirmation is in [`../reports/`](../reports/) and in the root
[`README.md`](../README.md).

## 1. Grid

Weights are constrained to

```
w = s · (n / 2^p) · k,      k ∈ {−2^p, …, −1, 1, …, 2^p},      step = n / 2^p
```

* `n` — half-range of the normalised grid,
* `p` — binary precision (integer ≥ 1),
* `s` — per-block scale (`s = absmax(block) / n`, block of 32 weights),
* `k` — integer grid index; the value `k = 0` is excluded.

The grid is symmetric and contains exactly `2^(p+1)` values:

```
levels = { −2^p, …, −1 } ∪ { 1, …, 2^p },    |levels| = 2^(p+1)
range  = [−n, −n/2^p] ∪ [+n/2^p, +n]
```

The gap between `k = −1` and `k = +1` is `2n/2^p = n/2^(p−1)`, i.e. twice the nominal
step. Crossing zero is a normal state of the grid, not an error: it simply consumes one
double step.

## 2. Why powers of two

Every grid point `k · n / 2^p` is a dyadic rational: with `n = 1` it is an integer
multiple of `2^-p` and therefore representable exactly in binary floating point. There
are no decimals such as `1/3`, `1/5`, `1/7`; encoding and decoding introduce no rounding,
and dequantization is a multiplication by a power of two combined with the block scale.

A symmetric int8 grid uses `step = absmax/127`; `127` is not a power of two, so the step
has an infinite binary expansion and every reconstruction is rounded. PoT-NZ trades the
zero level for an exact step.

## 3. Encoding: `p+1` bits, no dead code

```
encode:  c = k + 2^p        if k < 0
         c = k + 2^p − 1    if k > 0
decode:  k = c − 2^p        if c ≤ 2^p − 1
         k = c − 2^p + 1    if c ≥ 2^p
```

For `p = 4` the mapping is `k = −16 → 0`, `k = −1 → 15`, `k = +1 → 16`, `k = +16 → 31`;
all 32 codes are occupied. A grid with zero has `2^(p+1)+1` levels and needs `p+2` bits
(`ceil(log2(33)) = 6` for `p = 4`), or a zero-mask / entropy scheme.

## 4. Block scaling

Real weights are not bounded by `[−n, n]`, so a scale is factored out per block of 32:

```
s_b = absmax(w_b) / n
w   = s_b · (n/2^p) · k
```

Cost: `p+1` bits per code + 16 bits per 32 weights = `(p+1) + 0.5` bits per weight.
Because `s_b` equals the block maximum, the largest weight of each block always sits on
the grid boundary `|k| = 2^p`; the measured "clipping" share is therefore a structural
`1/32 = 3.125%`, independent of `p`. It is not a loss: the step is `absmax/(n·2^p)`, so
the boundary value is represented exactly.

## 5. QAT with straight-through gradients

```
q = k · step · s,       k = round(clamp(w / (step · s), −2^p, 2^p))
forward:  w + (q − w).detach()
```

The forward pass evaluates `q`; the backward pass is the identity with respect to `w`.
Optimizer state remains fp32 during training (QAT does not save memory at train time),
but the deployed model is compact.

Optional L1 regularization (`loss = CE + λ Σ|w|`) pushes weights toward the zero code of
the zero-including grid (variant B) — see §7.

## 6. Container formats

**SZP1** (dense):

```
"SZP1" | uint32 header length | JSON header | p+1-bit codes | fp16 block scales | fp32 rest
```

Eight 5-bit codes occupy exactly 5 bytes; the last byte is zero-padded.

**SZM1** (zero mask, variant B):

```
"SZM1" | uint32 header length | JSON header | 1-bit mask | p+1-bit codes of non-zeros |
        fp16 block scales | fp32 rest
```

Bit cost per weight:

```
SZP1:  (p+1) + 16/32
SZM1:  1 + (p+1)(1−f) + 16/32           (f = fraction of exact zeros)
```

Compression against fp32:

```
compression = 32 · (q + e) / (q · bits + 32 · e)
```

with `q` quantized and `e` fp32 parameters (embeddings, `lm_head`, LayerNorm, biases).
The ratio is governed by `q/(q+e)`: a large vocabulary relative to the body shrinks
compression unless the embedding is quantized as well.

## 7. Break-even thresholds

Sparse (`SZM1`) versus dense `p+1`-bit (`SZP1`):

```
1 + (p+1)(1−f) < p+1      ⟺    f > p/(p+1)
```

Concretely, against a 5-bit dense grid (`p = 4`):

| p | condition | note |
|---|---|---|
| 2 | any `f > 0` | mask is cheaper than a 3-bit code |
| 3 | `f > 0` (at the 5-bit target) | — |
| 4 | `f > 20%` | mask bit = 1/5 of a code |
| 5 | `f > 33%` | — |
| 6 | `f > 43%` | — |
| 7 | `f > 50%` | impractical for 8-bit grids |

Without renumbering the non-zero codes (keeping `p+2` bits) the classical threshold is

```
f + (1−f)(p+2) < p+1   ⟹   f > 1/(p+2) = 16.7%   (p = 4)
```

which is the "17%" quoted in the original note. Renumbering the non-zero codes into
`p+1` bits (using the zero-free encoding of §3) lowers the sparse format by one bit per
non-zero weight and raises the threshold against the 5-bit grid to 20%.

## 8. What is measured

For every run we report:

* validation perplexity and its ratio to the fp32 baseline of the same architecture,
* the share of weights exactly on the grid (`q == k·step·s`, must be 100%),
* the share of exact zeros (0% for variant A by construction, `f` for variant B),
* clipping (share at `|k| = 2^p`) and the relative quantization error
  `Σ|q−w| / Σ|w|`,
* the real size of the packed `.bin` and the compression against fp32.

Expected invariants: 100% on-grid, 0% zeros for A, structural clipping ≈ 1/32, and a
relative error that halves with each additional bit. All of them are confirmed on both
models (see the reports).
