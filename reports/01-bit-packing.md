# Report 1 — Bit packing and bit-width scaling

**Setup.** Character-level transformer, 826 368 parameters (786 432 quantized), vocabulary
65, `d = 128`, 4 heads, 4 layers, `d_ff = 512`, context 128; tiny-shakespeare, 90/10 split;
AdamW 3e-4 with cosine schedule; seed 42. The same architecture is trained for `p = 2…8`
and as an fp32 baseline (variant A). Quantized modules: `qkv`, `proj`, `fc1`, `fc2`;
embeddings, LayerNorm, biases and `lm_head` stay fp32.

## 1. Packing

Codes `k` occupy `p+1` bits and are written back to back, without byte alignment:

| codes | bits per code | bytes |
|---|---|---|
| 8 | 5 | **5** |
| 8 | 8 | 8 |
| 1000 | 3 | 375 |
| 1000 | 9 | 1125 |

The sparse end of the scale matters: at `p = 2` (3-bit codes) the packed stream is 2.67×
smaller than one byte per code; at `p = 7` (8 bits) packing gives nothing; at `p = 8`
(9 bits) it is *larger* than `uint8`.

Container `SZP1`: `"SZP1" | uint32 header length | JSON header | codes | fp16 block scales |
fp32 parameters`. The JSON header costs ≈ 6.8 KB for 53 tensors, i.e. 1% of a 700 KB file.

### Losslessness

```
codes: 786 432 bytes (uint8) -> 491 520 bytes (5-bit stream), 1.60x less
layers: 16, restored bit-exactly: 16/16   (with fp32 block scales)
weights from .bin: perplexity 7.819 vs 7.819 original (0.001% difference)
```

With fp16 scales the reconstruction deviates by ≤ 6.1e-05, which is the mantissa of the
scale only; codes themselves never lose information. A `float32` scale mode gives exact
weights at a cost of 0.5 extra bit per weight.

## 2. Bit-width scaling

60 epochs, fp32 perplexity 6.05:

| p | bits/weight | perplexity | ratio | clip | relative error | `.bin`, bytes | compression |
|---|---|---|---|---|---|---|---|
| fp32 | 32 | 6.05 | 1.000 | — | — | 3 310 329 | 1.00× |
| 4 | 5.5 | 5.96 | 0.985 | 3.6% | 5.25% | 707 233 | 4.67× |
| 5 | 6.5 | 5.96 | 0.985 | 3.4% | 2.45% | 805 538 | 4.10× |
| 6 | 7.5 | 5.93 | 0.980 | 3.2% | 1.18% | 903 843 | 3.66× |
| 7 | 8.5 | 5.96 | 0.984 | 3.2% | 0.58% | 1 002 148 | 3.30× |
| 8 | 9.5 | 5.95 | 0.984 | 3.2% | 0.29% | 1 100 453 | 3.00× |

30 epochs, fp32 perplexity 8.02:

| p | perplexity | ratio | compression |
|---|---|---|---|
| 2 | 7.79 | 0.972 | 6.47× |
| 3 | 8.01 | 0.999 | 5.43× |
| 4 | 7.82 | 0.976 | 4.67× |
| 5 | 8.02 | 1.000 | 4.10× |

## 3. Invariants

* **On-grid share:** 100% in every quantized run (`q == k·step·s` for all layers).
* **Zeros:** 0.000% for the zero-free grid, by construction.
* **Clipping:** 3.2–5.5% across all `p` — the structural `1/32` floor, not a quality loss.
  It does not decrease with `p`, because the block scale is the block maximum.
* **Relative error** `Σ|q−w|/Σ|w|` halves with each bit: 27.1% → 11.4% → 5.08% → 2.45% →
  1.18% → 0.58% → 0.29%. This is the direct empirical form of the `2^-p` step.

## 4. Discussion

The quality ceiling is reached at 5 bits. `p = 6` is nominally the best (5.93) but the
0.03 perplexity gain is within single-seed noise, while the file grows by 28%. Bits beyond
6 are pure overhead for this model. The aggressive end is surprisingly robust: 8 levels
(3 bits) still train to 7.79 against 8.02 fp32, although the relative error is 27% — the
QAT objective absorbs it.

---

# Отчёт 1 — упаковка бит и масштабируемость разрядности

**Постановка.** Char-level трансформер, 826 368 параметров (786 432 квантуется), словарь
65, `d = 128`, 4 головы, 4 слоя, `d_ff = 512`, контекст 128; tiny-shakespeare, split 90/10;
AdamW 3e-4 с cosine; seed 42. Та же архитектура обучается для `p = 2…8` и как fp32-база.
Квантуются `qkv`, `proj`, `fc1`, `fc2`; эмбеддинги, LayerNorm, bias и `lm_head` — fp32.

## 1. Упаковка

Коды `k` занимают `p+1` бит и пишутся подряд, без выравнивания на байт: 8 кодов по 5 бит —
ровно 5 байт; 1000 кодов по 3 бита — 375 байт; 1000 по 9 бит — 1125 байт. На 8 битах
упаковка ничего не даёт, на 9 — делает файл больше `uint8`.

Контейнер `SZP1`: `"SZP1" | uint32 | JSON | коды | масштабы fp16 | fp32-параметры`.
JSON-заголовок ≈ 6.8 КБ на 53 тензора (≈1% файла 700 КБ).

### Проверка без потерь

```
коды: 786 432 байт (uint8) -> 491 520 байт (поток 5 бит), экономия 1.60x
слоёв: 16, побитово восстановлено: 16/16   (fp32-масштабы)
perplexity из .bin: 7.819 против 7.819 (разница 0.001%)
```

При fp16-масштабах расхождение ≤ 6.1e-05 — это только мантисса масштаба, коды не теряются.

## 2. Масштабируемость разрядности

60 эпох, fp32 = 6.05:

| p | бит/вес | perplexity | к fp32 | клип | отн. ошибка | `.bin`, байт | сжатие |
|---|---|---|---|---|---|---|---|
| fp32 | 32 | 6.05 | 1.000 | — | — | 3 310 329 | 1.00× |
| 4 | 5.5 | 5.96 | 0.985 | 3.6% | 5.25% | 707 233 | 4.67× |
| 5 | 6.5 | 5.96 | 0.985 | 3.4% | 2.45% | 805 538 | 4.10× |
| 6 | 7.5 | 5.93 | 0.980 | 3.2% | 1.18% | 903 843 | 3.66× |
| 7 | 8.5 | 5.96 | 0.984 | 3.2% | 0.58% | 1 002 148 | 3.30× |
| 8 | 9.5 | 5.95 | 0.984 | 3.2% | 0.29% | 1 100 453 | 3.00× |

30 эпох, fp32 = 8.02: `p = 2` → 7.79 (6.47×), `p = 3` → 8.01 (5.43×), `p = 4` → 7.82 (4.67×),
`p = 5` → 8.02 (4.10×).

## 3. Инварианты

* **На сетке:** 100% во всех квантованных прогонах.
* **Нули:** 0.000% для сетки без нуля — по построению.
* **Клип:** 3.2–5.5% при любом `p` — структурный пол `1/32`, а не потеря качества.
* **Относительная ошибка** падает вдвое на бит: 27.1% → 11.4% → 5.08% → 2.45% → 1.18% →
  0.58% → 0.29% — прямое подтверждение шага `2^-p`.

## 4. Обсуждение

Потолок качества достигается на 5 битах. `p = 6` формально лучший (5.93), но выигрыш
0.03 — в пределах шума одного seed, а файл на 28% больше. Биты выше 6 — чистые накладные
расходы. Агрессивный край устойчив: 8 уровней (3 бита) дают 7.79 против 8.02 у fp32, хотя
относительная ошибка 27%.
