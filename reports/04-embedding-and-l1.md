# Report 4 — Embedding quantization and L1 sparsity

Two experiments on the question left open by Reports 1–3: (a) is it worth quantizing the
token embedding and `lm_head`, and (b) can L1 regularization produce enough exact zeros
to make the zero-mask container pay off.

## Part A — Quantized embedding (medium model, 7.29M)

`QuantEmbedding` applies the same zero-free grid to the embedding weight; with tied
embeddings a single `QuantHead` reuses that weight for `lm_head`, so it is quantized once.
No other change to the architecture or the training protocol.

| model | p | bits/weight | perplexity (fp32) | `.bin`, bytes | compression |
|---|---|---|---|---|---|
| A fp32 | — | 32 | 258.8 | 29 158 216 | 1.00× |
| C p=4, embedding fp32 | 4 | 5.5 | 254.8 | 18 739 941 | 1.56× |
| C p=7, embedding fp32 | 7 | 8.5 | 235.4 | 19 919 619 | 1.46× |
| C p=7 + embedding | 7 | 8.5 | 241.1 | 7 887 735 | **3.70×** |
| C p=4 + embedding | 4 | 5.5 | 239.3 | 5 172 084 | **5.64×** |

* Quality: 241.1 (p=7) and 239.3 (p=4) against 258.8 fp32 — ratios 0.931 and 0.925.
  Embedding quantization costs 2.4% relative to the same model without it (241.1 vs 235.4),
  which is inside the 1.1× acceptance threshold.
* Size: 19.9 MB → 7.9 MB at 8 bits, and 18.7 MB → 5.17 MB at 5 bits. The 3.70× factor is
  exactly `32/8.5`; at 5 bits the measured 5.64× matches the predicted 5.82× up to the
  JSON header.

### Embedding weight range after QAT

| model | embedding | max \|w\| | share \|w\| > 1 | grid clip | zeros |
|---|---|---|---|---|---|
| A fp32 | fp32 | 0.185 | 0.0000% | — | — |
| C p=7 | fp32 | 0.359 | 0.0000% | — | — |
| C p=7 + embedding | QAT | 0.342 | **0.0000%** | 3.20% | 0.000% |
| C p=4 + embedding | QAT | 0.367 | **0.0000%** | 3.81% | 0.000% |

Weights stay well inside `[−1, 1]` before and after QAT (maximum 0.367 against a limit of
1.0); the 3.2–3.8% clipping is the structural block-maximum effect described in the theory,
and the per-block scale `|u|/n = 1.0` absorbs any range by construction. A large vocabulary
does not push weights out of range.

## Part B — L1 regularization (character model, variant B)

Loss: `CE + λ·Σ|w|` over the quantized weights, zero-including grid, 60 epochs.
Baselines of the same series: fp32 4.80, zero-free 5-bit 4.80.

**p = 7 (8-bit codes)**

| λ | zeros | perplexity | ratio | dense `.bin` | SZM1 `.bin` | bits/weight |
|---|---|---|---|---|---|---|
| 0 | 0.73% | 4.79 | 0.998 | 1 100 454 | 1 095 438 | 9.44 |
| 1e-5 | 5.90% | 4.97 | 1.035 | 1 100 454 | 1 054 779 | 9.03 |
| 5e-5 | 36.06% | 6.29 | 1.310 | 1 100 454 | 817 568 | 6.61 |
| 2e-4 | 13.83% | 9.62 | 2.004 | 1 100 454 | 992 405 | 8.39 |
| 1e-3 | 0.81% | 11.75 | 2.448 | 1 100 454 | 1 094 843 | 9.44 |

At 8 bits the mask never pays: even 36% zeros leave 817 KB against 707 KB of the dense
5-bit grid (+16%).

**p = 4 (5-bit codes, comparable with the dense baseline)**

| λ | zeros | perplexity | ratio | dense `.bin` | SZM1 `.bin` | bits/weight | vs dense 5-bit (707 233) |
|---|---|---|---|---|---|---|---|
| 0 | 5.91% | 4.79 | 0.998 | 805 539 | 777 243 | 6.20 | +9.9% |
| 1e-5 | 18.41% | 4.99 | 1.039 | 805 539 | 715 778 | 5.58 | +1.2% |
| **2e-5** | **32.01%** | **5.25** | **1.094** | 805 539 | **648 968** | 4.90 | **−8.2%** |
| 3e-5 | 39.33% | 5.66 | 1.179 | 805 539 | 612 948 | 4.53 | −13.3% |
| 5e-5 | 43.64% | 6.20 | 1.292 | 805 539 | 591 802 | 4.32 | −16.3% |

**Working point:** `λ = 2e-5` — 32% exact zeros, 8.2% smaller file than the dense 5-bit
grid, perplexity 5.25 against 4.80 (1.094, inside the 1.1 threshold). Larger λ buys size at
quality cost; smaller λ (1e-5, 18.4% zeros) lands just below the 20% break-even and loses.

**Why the zero share is not monotonic in λ.** The block scale equals the block maximum, so
L1 shrinks the whole block, `absmax` falls with it, the step `absmax/2^p` shrinks too, and
the zero window `|w| < step/2` stays narrow. Zeros appear only when a weight becomes small
*relative to its own block*. Hence very large λ (2e-4, 1e-3) collapses the weights as a
whole, the zero share drops again and perplexity degrades (9.6 and 11.8). Coarse grids
(p ≤ 4) are therefore the natural habitat of the zero-mask container.

## Conclusions

1. Quantizing the embedding and `lm_head` is the single largest size win: 1.56× → 5.64×
   (p = 4) at quality not worse than fp32. Recommended.
2. L1 pays off only on coarse grids and only at a working point with `f > 20%`
   (`λ = 2e-5`, p = 4): −8.2% file for +9.4% perplexity. At p ≥ 6 the mask is dead weight.
3. The two techniques compose: a 5-bit zero-free grid for the body and embedding, plus an
   optional zero-mask variant when explicit sparsity is available.

---

# Отчёт 4 — квантование эмбеддинга и L1-разреженность

Два эксперимента по вопросам, оставленным отчётами 1–3: (а) стоит ли квантовать
токенный эмбеддинг и `lm_head`, (б) даёт ли L1-регуляризация достаточно точных нулей,
чтобы окупился контейнер с маской.

## Часть A — квантованный эмбеддинг (средняя модель, 7.29M)

`QuantEmbedding` применяет ту же сетку без нуля к весу эмбеддинга; при tied embeddings
`QuantHead` переиспользует этот вес для `lm_head`, поэтому квантование однократное.

| модель | p | бит/вес | perplexity (fp32) | `.bin`, байт | сжатие |
|---|---|---|---|---|---|
| A fp32 | — | 32 | 258.8 | 29 158 216 | 1.00× |
| C p=4, эмбеддинг fp32 | 4 | 5.5 | 254.8 | 18 739 941 | 1.56× |
| C p=7, эмбеддинг fp32 | 7 | 8.5 | 235.4 | 19 919 619 | 1.46× |
| C p=7 + эмбеддинг | 7 | 8.5 | 241.1 | 7 887 735 | **3.70×** |
| C p=4 + эмбеддинг | 4 | 5.5 | 239.3 | 5 172 084 | **5.64×** |

Качество: 241.1 (p=7) и 239.3 (p=4) против 258.8 у fp32 (отношения 0.931 и 0.925).
Квантование эмбеддинга стоит 2.4% относительно той же модели без него (241.1 против
235.4) — в пределах порога 1.1×. Размер: 19.9 → 7.9 МБ на 8 битах и 18.7 → 5.17 МБ на
5 битах; 3.70× — это ровно `32/8.5`, а 5.64× совпадает с прогнозом 5.82× с точностью до
JSON-заголовка.

### Диапазон весов эмбеддинга после QAT

| модель | эмбеддинг | max \|w\| | доля \|w\| > 1 | клип | нули |
|---|---|---|---|---|---|
| A fp32 | fp32 | 0.185 | 0.0000% | — | — |
| C p=7 | fp32 | 0.359 | 0.0000% | — | — |
| C p=7 + эмбеддинг | QAT | 0.342 | **0.0000%** | 3.20% | 0.000% |
| C p=4 + эмбеддинг | QAT | 0.367 | **0.0000%** | 3.81% | 0.000% |

Веса остаются внутри `[−1, 1]` до и после QAT (максимум 0.367 при пороге 1.0); клип
3.2–3.8% — структурный эффект максимума блока, а масштаб `|u|/n = 1.0` поглощает любой
размах по построению. Большой словарь не выводит веса за диапазон.

## Часть B — L1-регуляризация (char-модель, вариант B)

`loss = CE + λ·Σ|w|` по квантуемым весам, сетка с нулём, 60 эпох. Базы той же серии:
fp32 4.80, сетка без нуля 5 бит 4.80.

**p = 7 (8-битные коды)**

| λ | нули | perplexity | к fp32 | плотный `.bin` | SZM1 `.bin` | бит/вес |
|---|---|---|---|---|---|---|
| 0 | 0.73% | 4.79 | 0.998 | 1 100 454 | 1 095 438 | 9.44 |
| 1e-5 | 5.90% | 4.97 | 1.035 | 1 100 454 | 1 054 779 | 9.03 |
| 5e-5 | 36.06% | 6.29 | 1.310 | 1 100 454 | 817 568 | 6.61 |
| 2e-4 | 13.83% | 9.62 | 2.004 | 1 100 454 | 992 405 | 8.39 |
| 1e-3 | 0.81% | 11.75 | 2.448 | 1 100 454 | 1 094 843 | 9.44 |

На 8 битах маска не окупается никогда: даже 36% нулей дают 817 КБ против 707 КБ плотной
5-битной сетки (+16%).

**p = 4 (5-битные коды, сравнимо с плотной базой)**

| λ | нули | perplexity | к fp32 | плотный `.bin` | SZM1 `.bin` | бит/вес | к плотной 5-битной |
|---|---|---|---|---|---|---|---|
| 0 | 5.91% | 4.79 | 0.998 | 805 539 | 777 243 | 6.20 | +9.9% |
| 1e-5 | 18.41% | 4.99 | 1.039 | 805 539 | 715 778 | 5.58 | +1.2% |
| **2e-5** | **32.01%** | **5.25** | **1.094** | 805 539 | **648 968** | 4.90 | **−8.2%** |
| 3e-5 | 39.33% | 5.66 | 1.179 | 805 539 | 612 948 | 4.53 | −13.3% |
| 5e-5 | 43.64% | 6.20 | 1.292 | 805 539 | 591 802 | 4.32 | −16.3% |

**Рабочая точка:** `λ = 2e-5` — 32% точных нулей, файл на 8.2% меньше плотной 5-битной
сетки, perplexity 5.25 против 4.80 (1.094, в пределах порога 1.1). Большие λ покупают
размер ценой качества; меньшее λ (1e-5, 18.4% нулей) не дотягивает до порога 20% и
проигрывает.

**Почему доля нулей немонотонна по λ.** Масштаб блока равен максимуму блока: L1 сжимает
блок целиком, `absmax` падает вместе с ним, шаг `absmax/2^p` тоже, и окно нуля
`|w| < шаг/2` остаётся узким. Ноль появляется, только когда вес мал *относительно своего
блока*. Поэтому очень большие λ (2e-4, 1e-3) просаживают веса целиком, доля нулей снова
падает, а perplexity деградирует (9.6 и 11.8). Грубые сетки (p ≤ 4) — естественная среда
для контейнера с маской.

## Выводы

1. Квантование эмбеддинга и `lm_head` — главный выигрыш по размеру: 1.56× → 5.64× (p=4)
   при качестве не хуже fp32. Рекомендуется.
2. L1 оправдана только на грубых сетках и только в рабочей точке `f > 20%`
   (`λ = 2e-5`, p=4): −8.2% файла за +9.4% perplexity. При p ≥ 6 маска — мёртвый груз.
3. Техники совместимы: 5-битная сетка без нуля для тела и эмбеддинга плюс опциональный
   zero-mask вариант, когда есть явная разреженность.
