# Report 3 — Medium model (BPE-16k, 7.29M) and what compression really depends on

**Setup.** Decoder-only transformer, `d = 256`, 8 heads, 4 layers, `d_ff = 1024`,
context 128, token embedding tied with `lm_head`; 7 288 320 parameters, of which
3 145 728 are quantized (attention/FFN). The tokenizer is a byte-level BPE with a
16 000-token vocabulary trained on the train split of tiny-shakespeare; the corpus yields
305 035 tokens. The optimizer is AdamW 3e-4 with cosine schedule, batch 32, 60 epochs,
seed 42. Two regularization regimes: dropout 0.1 / weight decay 0.01 and dropout 0.2 /
weight decay 0.1.

## 1. Tokenization

| metric | value |
|---|---|
| vocabulary | 16 000 |
| tokens | 305 035 (1 115 394 characters) |
| characters per token | 3.66 |
| train / validation tokens | 271 316 / 33 719 |
| occurrences per vocabulary entry | 16.96 |
| validation tokens unseen in training | 5.23% (1 765 of 33 719) |

A 16 000-token vocabulary on 1.1 MB of text is not supported by the data: rare tokens
occur once or twice, and 5% of validation tokens never appear in training. The model
therefore overfits quickly — the corpus, not the quantizer, is the binding constraint.

## 2. Training results (perplexity, best checkpoint / after 60 epochs)

| run | dropout 0.1, wd 0.01 | dropout 0.2, wd 0.1 |
|---|---|---|
| A fp32 | 258.8 (ep 8) / 557.2 | 249.9 (ep 12) / 386.8 |
| C p=4 | 254.8 (ep 26) / 424.4 | 256.5 (ep 20) / 350.6 |
| C p=5 | 251.6 (ep 20) / 442.7 | 256.4 (ep 24) / 347.8 |

Ratios to fp32: best checkpoint 0.985 / 0.972 (dropout 0.1) and 1.026 / 1.026
(dropout 0.2); after 60 epochs 0.762 / 0.795 and 0.906 / 0.899. The hypothesis
"QAT perplexity ≤ 1.1 × fp32" holds in every configuration. At the end of training the
quantized models overfit *less* than fp32 — the grid acts as a regularizer.

Training time: fp32 189–193 s, QAT 225–230 s, i.e. **+18–22%** for quantizing weights at
every step.

## 3. Compression

| model | `.bin`, bytes | compression |
|---|---|---|
| A fp32 | 29 158 216 | 1.00× |
| C p=4 | 18 739 941 | 1.56× |
| C p=5 | 19 133 158 | 1.52× |
| all parameters quantized (hypothetical, 5.5 bit) | 5 010 720 | 5.82× |

Compression fell from 4.67× (character model) to 1.56×. The reason is the parameter mix:
4 142 592 of the 7 288 320 parameters (56.8%) are the fp32 embedding (16 000 × 256,
shared with `lm_head`) plus positional embeddings and LayerNorm. The ratio is governed by
`q/(q+e)`:

```
compression = 32 · (q + e) / (q · bits + 32 · e)
```

| configuration | q | e | compression |
|---|---|---|---|
| this model (4 layers, BPE-16k) | 3.1M | 4.14M | 1.56× |
| 8 layers, same vocabulary | 6.3M | 4.14M | 2.00× |
| 12 layers, same vocabulary | 9.4M | 4.14M | 2.36× |
| 24 layers, same vocabulary | 18.9M | 4.14M | 3.12× |
| embedding also quantized | 7.3M | 0.0M | 5.82× |
| real 7B LLM (vocab 32k, d=4096) | 6.7B | 0.13B | 5.33× |

## 4. Discussion

Growing the parameter count does not by itself improve compression — the ratio of body to
embedding does. For a small model with a large vocabulary the embedding dominates unless
it is quantized too (Report 4). At the size of a real LLM the embedding is ~2% of the
parameters, and the formula returns ≈5.3×.

---

# Отчёт 3 — средняя модель (BPE-16k, 7.29M) и от чего реально зависит сжатие

**Постановка.** Decoder-only трансформер, `d = 256`, 8 голов, 4 слоя, `d_ff = 1024`,
контекст 128, эмбеддинг связан с `lm_head`; 7 288 320 параметров, из них 3 145 728
квантуется. Токенизатор — byte-level BPE на 16 000 токенов, обучен на train-части
tiny-shakespeare; корпус даёт 305 035 токенов. AdamW 3e-4, cosine, batch 32, 60 эпох,
seed 42. Два режима регуляризации: dropout 0.1 / wd 0.01 и dropout 0.2 / wd 0.1.

## 1. Токенизация

| показатель | значение |
|---|---|
| словарь | 16 000 |
| токенов | 305 035 (1 115 394 символа) |
| символов на токен | 3.66 |
| train / val токенов | 271 316 / 33 719 |
| вхождений на запись словаря | 16.96 |
| токенов валидации, не встречавшихся в train | 5.23% (1 765 из 33 719) |

Словарь на 16 000 токенов не обеспечен данными: редкие токены встречаются один-два раза,
5% валидации — незнакомые токены. Модель быстро переобучается: узкое место — корпус, а не
квантователь.

## 2. Результаты обучения (perplexity: лучший чекпоинт / после 60 эпох)

| прогон | dropout 0.1, wd 0.01 | dropout 0.2, wd 0.1 |
|---|---|---|
| A fp32 | 258.8 (эп. 8) / 557.2 | 249.9 (эп. 12) / 386.8 |
| C p=4 | 254.8 (эп. 26) / 424.4 | 256.5 (эп. 20) / 350.6 |
| C p=5 | 251.6 (эп. 20) / 442.7 | 256.4 (эп. 24) / 347.8 |

Отношения к fp32: лучший чекпоинт 0.985 / 0.972 и 1.026 / 1.026; после 60 эпох
0.762 / 0.795 и 0.906 / 0.899. Гипотеза «perplexity QAT ≤ 1.1 × fp32» выполняется во всех
конфигурациях. К концу обучения квантованные модели переобучаются *меньше* fp32 — сетка
работает как регуляризатор.

Время: fp32 189–193 с, QAT 225–230 с, то есть **+18–22%** за квантование весов на каждом шаге.

## 3. Сжатие

| модель | `.bin`, байт | сжатие |
|---|---|---|
| A fp32 | 29 158 216 | 1.00× |
| C p=4 | 18 739 941 | 1.56× |
| C p=5 | 19 133 158 | 1.52× |
| все параметры квантованы (гипотетически, 5.5 бита) | 5 010 720 | 5.82× |

Сжатие упало с 4.67× (char-модель) до 1.56×. Причина — состав параметров: 4 142 592 из
7 288 320 (56.8%) — fp32-эмбеддинг (16 000 × 256, общий с `lm_head`), плюс позиционные
эмбеддинги и LayerNorm. Отношение определяется `q/(q+e)`:

```
сжатие = 32 · (q + e) / (q · биты + 32 · e)
```

| конфигурация | q | e | сжатие |
|---|---|---|---|
| эта модель (4 слоя, BPE-16k) | 3.1M | 4.14M | 1.56× |
| 8 слоёв, тот же словарь | 6.3M | 4.14M | 2.00× |
| 12 слоёв | 9.4M | 4.14M | 2.36× |
| 24 слоя | 18.9M | 4.14M | 3.12× |
| эмбеддинг тоже квантован | 7.3M | 0.0M | 5.82× |
| реальная LLM 7B (vocab 32k, d=4096) | 6.7B | 0.13B | 5.33× |

## 4. Обсуждение

Рост числа параметров сам по себе сжатие не улучшает — улучшает его отношение «тело /
эмбеддинг». Для маленькой модели с большим словарём эмбеддинг доминирует, если его не
квантовать (отчёт 4). На масштабе реальной LLM эмбеддинг — около 2% параметров, и формула
даёт ≈5.3×.
