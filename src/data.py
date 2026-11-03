"""Данные: char-level токенизация, 90/10 split, батчи (план, раздел 3)."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field

import torch

DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class CharTokenizer:
    kind = "char"

    stoi: dict[str, int] = field(default_factory=dict)
    itos: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        return cls(stoi={c: i for i, c in enumerate(chars)}, itos=chars)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        missing = sorted(set(text) - set(self.stoi))
        if missing:
            raise ValueError(f"символы вне словаря: {missing[:10]!r}")
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)

    def to_json(self) -> str:
        return json.dumps({"kind": "char", "stoi": self.stoi, "itos": self.itos},
                          ensure_ascii=False)

    @classmethod
    def from_json(cls, blob: str) -> "CharTokenizer":
        obj = json.loads(blob)
        return cls(stoi=obj["stoi"], itos=obj["itos"])


class BPETokenizer:
    """Байтовый BPE (HuggingFace tokenizers): неизвестных токенов нет."""

    kind = "bpe"

    def __init__(self, tok) -> None:
        self.tok = tok

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tok.encode(text).ids, dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tok.decode([int(i) for i in ids])

    def save(self, path: str) -> None:
        self.tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))


def train_bpe(text: str, vocab_size: int, out_path: str) -> BPETokenizer:
    from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders
    from tokenizers.models import BPE

    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator([text], trainer=trainer)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tok.save(out_path)
    return BPETokenizer(tok)


def save_tokenizer(tok, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    if isinstance(tok, BPETokenizer):
        with open(os.path.join(run_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"kind": "bpe"}, f)
        tok.save(os.path.join(run_dir, "bpe_tokenizer.json"))
    else:
        with open(os.path.join(run_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write(tok.to_json())


def load_tokenizer(run_dir: str):
    path = os.path.join(run_dir, "tokenizer.json")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.loads(f.read())
    if obj.get("kind") == "bpe":
        return BPETokenizer.load(os.path.join(run_dir, "bpe_tokenizer.json"))
    return CharTokenizer(stoi=obj["stoi"], itos=obj["itos"])


def load_text(path: str, url: str | None = DEFAULT_URL) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if not url:
        raise FileNotFoundError(f"нет файла данных: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[data] скачиваю корпус: {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read().decode("utf-8")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(raw)
    return raw


@dataclass
class DataBundle:
    tok: CharTokenizer
    train: torch.Tensor
    val: torch.Tensor
    block_size: int

    def steps_per_epoch(self, batch_size: int) -> int:
        return max(1, (len(self.train) - 1) // (batch_size * self.block_size))


def build_tokenizer(cfg: dict, text: str):
    tcfg = cfg.get("tokenizer") or {"type": "char"}
    kind = tcfg.get("type", "char")
    if kind == "char":
        return CharTokenizer.build(text)
    if kind == "bpe":
        path = tcfg.get("file", os.path.join("data", f"bpe{tcfg['vocab_size']}", "tokenizer.json"))
        if os.path.exists(path):
            return BPETokenizer.load(path)
        print(f"[data] обучаю BPE (vocab={tcfg['vocab_size']}) -> {path}")
        return train_bpe(text, int(tcfg["vocab_size"]), path)
    raise ValueError(f"неизвестный тип токенизатора: {kind!r}")


def make_bundle(cfg: dict) -> DataBundle:
    text = load_text(cfg["path"], cfg.get("url", DEFAULT_URL))
    n_val = int(len(text) * float(cfg.get("val_fraction", 0.1)))
    train_text, val_text = text[: len(text) - n_val], text[len(text) - n_val :]
    tok = build_tokenizer(cfg, train_text)
    train = tok.encode(train_text)
    val = tok.encode(val_text)
    return DataBundle(tok=tok, train=train, val=val, block_size=int(cfg["block_size"]))


def tokenizer_meta(cfg: dict) -> dict:
    tcfg = cfg.get("tokenizer") or {"type": "char"}
    return {"kind": tcfg.get("type", "char"), "vocab_size": tcfg.get("vocab_size")}


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device, generator=None):
    if len(data) <= block_size + 1:
        raise ValueError("слишком короткий корпус для указанного context")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def val_loader(data: torch.Tensor, batch_size: int, block_size: int, device, max_batches: int | None = None):
    """Детерминированные непересекающиеся окна валидации, покрывающие весь split."""
    n = (len(data) - 1) // block_size
    xs, ys = [], []
    for i in range(0, n * block_size, batch_size * block_size):
        idxs = list(range(i, min(i + batch_size * block_size, n * block_size), block_size))
        if not idxs:
            continue
        x = torch.stack([data[j : j + block_size] for j in idxs])
        y = torch.stack([data[j + 1 : j + 1 + block_size] for j in idxs])
        xs.append(x.to(device))
        ys.append(y.to(device))
        if max_batches and len(xs) >= max_batches:
            break
    return list(zip(xs, ys))
