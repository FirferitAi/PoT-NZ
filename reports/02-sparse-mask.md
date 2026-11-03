# Report 2 — Zero-mask container (SZM1) and the break-even fraction

**Setup.** Character model from Report 1, variant B: the grid *includes* zero
(`skip_zero = false`, 33 levels at `p = 4`), so the trained model contains exact zeros.
Weights: `runs/B_p4_zero`, 826 368 parameters, 45 811 exact zeros (5.83%).

## 1. Format

```
"SZM1" | uint32 header length | JSON header |
        1-bit mask per weight (1 = non-zero)  |
        p+1-bit codes of the non-zero weights |
        fp16 block scales                     |
        fp32 parameters
```

The non-zero codes are **renumbered** into the zero-free encoding (Report 1, §2), so a
non-zero weight costs `p+1` bits instead of `p+2`.

## 2. Losslessness

```
format: SZM1, p=4, bits per non-zero code: 5
sparsity: 45 811 zeros of 786 432 weights (5.83%)
layers: 16, restored bit-exactly (fp32 scales): 16/16
zeros in place: 16/16 layers (45 811 total)
deviation with fp16 scales: 6.09e-05
perplexity from sparse .bin: 7.931 (original 7.931)
```

Zeros are placed exactly at `0.0`; non-zeros are exact up to the fp16 scale mantissa.

## 3. Sizes

| file | bytes | composition |
|---|---|---|
| dense `.bin` (6-bit codes) | 805 489 | codes 589 824 + scales 49 152 + fp32 159 744 + header 7 497 |
| **sparse `.bin` (1-bit mask + 5-bit)** | **777 600** | mask 98 304 + codes 462 895 + scales 49 152 + fp32 159 744 + header 7 497 |
| zero-free 5-bit grid, same architecture | 707 233 | codes 491 520 + scales 49 152 + fp32 159 744 + header 6 817 |

Sparse vs dense-B: **−3.46%**. Sparse vs zero-free 5-bit: **+9.96%** — at 5.83% zeros
the mask bit is more expensive than the code bits it saves.

## 4. Break-even

Bit cost per weight, `f` = fraction of zeros:

```
SZP1 (dense p+1):        p+1
SZM1 (mask):             1 + (p+1)(1−f)
dense with zero (p+2):   p+2
```

Against the dense `p+1` grid: `f > p/(p+1)`; against a 5-bit target:

| p | threshold vs 5-bit | at f = 5.83% |
|---|---|---|
| 3 | any `f` | wins |
| 4 | `f > 20%` | loses (+9.96%) |
| 5 | `f > 33%` | loses |
| 7 | `f > 50%` | loses |

Without renumbering (`p+2`-bit codes) the classical threshold is `f > 1/(p+2) = 16.7%` —
the "17%" of the original note. Empirical payload sizes (no header/fp32), bytes:

| f | sparse | dense B (6 bit) | dense C (5 bit) |
|---|---|---|---|
| 0.00% | 638 976 | 638 976 | 540 672 |
| 5.83% (measured) | 610 321 | 638 976 | 540 672 |
| 20.00% | 540 673 | 638 976 | 540 672 |
| 40.00% | 442 368 | 638 976 | 540 672 |

## 5. Discussion

The mask pays for itself only when zeros exceed 20% at `p = 4` (1 bit of mask against
5 bits of code, minus the saved codes). The measured 5.83% is far below that, so for
vanilla QAT the zero-free grid is strictly better. Zero-mask storage becomes interesting
only with explicit sparsity pressure (L1, Report 4) or block-structured masks.

---

# Отчёт 2 — контейнер с маской нулей (SZM1) и порог окупаемости

**Постановка.** Char-модель из отчёта 1, вариант B: сетка *содержит* ноль
(`skip_zero = false`, 33 значения при `p = 4`). Файл `runs/B_p4_zero`, 826 368 параметров,
45 811 точных нулей (5.83%).

## 1. Формат

```
"SZM1" | uint32 | JSON | маска 1 бит на вес (1 = ненулевой) |
        коды ненулевых (p+1 бит) | масштабы fp16 | fp32-параметры
```

Коды ненулевых **перенумерованы** в кодирование без нуля (отчёт 1, §2): ненулевой вес
стоит `p+1` бит вместо `p+2`.

## 2. Без потерь

```
формат: SZM1, p=4, бит на ненулевой код: 5
разреженность: 45 811 нулей из 786 432 (5.83%)
слоёв: 16, побитово восстановлено: 16/16
нули на своих местах: 16/16 слоёв (всего 45 811)
perplexity из разреженного .bin: 7.931 (исходная 7.931)
```

## 3. Размеры

| файл | байт |
|---|---|
| плотный `.bin` (6-битные коды) | 805 489 |
| **разреженный `.bin` (1 бит маски + 5 бит)** | **777 600** (−3.46%) |
| сетка без нуля, 5 бит, та же архитектура | 707 233 (+9.96% к разреженному) |

## 4. Порог окупаемости

Стоимость на вес: `p+1` (плотный), `1 + (p+1)(1−f)` (маска), `p+2` (плотный с нулём).
Против 5-битной сетки: `f > 20%` при `p = 4`; `f > 50%` при `p = 7`; при `p = 3` —
любая ненулевая доля. Без перенумерации кодов классический порог `1/(p+2) = 16.7%` —
это и есть «17%» исходной записи.

## 5. Обсуждение

Маска окупается только при доле нулей выше 20% (при `p = 4`). Измеренные 5.83% далеки от
этого, поэтому для обычного QAT сетка без нуля строго лучше. Маска становится
осмысленной только при явном давлении на разреженность (L1, отчёт 4) или блочных масках.
