"""Обучение одной модели (A/B/C) с QAT и полным логированием (план, разделы 5-7)."""
from __future__ import annotations

import sys

try:  # консоль Windows по умолчанию cp1251 — иначе "×" и "→" ломают print
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch

import data as data_lib
from model import ModelConfig, TinyGPT
from quant import collect_stats, grad_health


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_quant(cfg: dict, variant: str) -> dict | None:
    """A - fp32, B - сетка с нулём, C - сетка без нуля (теория)."""
    if variant == "A":
        return None
    q = dict(cfg["quant"])
    q["skip_zero"] = variant == "C"
    q.pop("_comment", None)
    return q


def lr_at(step: int, total: int, base_lr: float, warmup_frac: float, min_ratio: float) -> float:
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cos)


@torch.no_grad()
def evaluate(model: TinyGPT, loader) -> tuple[float, float]:
    model.eval()
    total_loss, total_tok = 0.0, 0
    for x, y in loader:
        _, loss = model(x, y)
        n = y.numel()
        total_loss += loss.item() * n
        total_tok += n
    model.train()
    mean = total_loss / max(total_tok, 1)
    return mean, math.exp(mean)


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
def train_one(cfg: dict, variant: str, tag: str, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    device = pick_device(cfg.get("device", "auto"))
    seed = int(overrides.get("seed", cfg["seed"]))
    set_seed(seed)

    tcfg = cfg["train"]
    epochs = int(overrides.get("epochs", tcfg["epochs"]))
    batch_size = int(tcfg["batch_size"])
    block_size = int(cfg["data"]["block_size"])

    bundle = data_lib.make_bundle({**cfg["data"], "block_size": block_size})
    quant = resolve_quant(cfg, variant)

    extra = {k: v for k, v in cfg["model"].items()
             if k in ModelConfig.__dataclass_fields__ and k not in ("vocab_size", "quant")}
    mcfg = ModelConfig(vocab_size=bundle.tok.vocab_size, quant=quant, **extra)
    model = TinyGPT(mcfg).to(device)
    model.train()

    run_dir = os.path.join(cfg["out_dir"], tag)
    os.makedirs(run_dir, exist_ok=True)
    data_lib.save_tokenizer(bundle.tok, run_dir)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(tcfg["weight_decay"])},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(tcfg["lr"]), betas=(0.9, 0.95),
    )

    steps_per_epoch = bundle.steps_per_epoch(batch_size)
    total_steps = epochs * steps_per_epoch
    gen = torch.Generator(device="cpu").manual_seed(seed)

    qmods = model.quant_linears()
    l1_lambda = float(tcfg.get("l1_lambda", 0.0) or 0.0)
    l1_mode = tcfg.get("l1_mode", "sum")
    if l1_lambda:
        print(f"[{tag}] L1: lambda={l1_lambda:g} ({l1_mode}) по {len(qmods)} тензорам")
    info = {
        "tag": tag,
        "variant": variant,
        "device": str(device),
        "torch": torch.__version__,
        "seed": seed,
        "params_total": model.n_params(),
        "params_quant": model.n_quant_params(),
        "quant": quant,
        "model": asdict(mcfg) | {"quant": None},
        "train": tcfg,
        "data": {"path": cfg["data"]["path"], "tokenizer": data_lib.tokenizer_meta(cfg["data"]),
                 "chars": int(len(bundle.train) + len(bundle.val)),
                 "train": int(len(bundle.train)), "val": int(len(bundle.val)),
                 "vocab": bundle.tok.vocab_size},
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
    write_json(os.path.join(run_dir, "run.json"), info)

    step_csv = open(os.path.join(run_dir, "train_log.csv"), "w", newline="", encoding="utf-8")
    w_step = csv.writer(step_csv)
    w_step.writerow(["step", "epoch", "lr", "train_loss", "grad_norm", "ste_grad_norm", "l1_penalty", "elapsed_s"])
    ep_csv = open(os.path.join(run_dir, "epoch_log.csv"), "w", newline="", encoding="utf-8")
    w_ep = csv.writer(ep_csv)
    w_ep.writerow(["epoch", "step", "train_loss_epoch", "val_loss", "val_ppl", "lr", "elapsed_s"])
    diag_path = os.path.join(run_dir, "diag.jsonl")
    diag_f = open(diag_path, "w", encoding="utf-8")

    log_iv = int(tcfg.get("log_interval_steps", 20))
    diag_iv = int(tcfg.get("diag_interval_steps", 200))
    hist = []
    best = {"val_loss": float("inf"), "step": 0}
    t0 = time.time()
    gstep = 0
    running = 0.0
    running_n = 0

    try:
        for epoch in range(1, epochs + 1):
            ep_loss, ep_n = 0.0, 0
            for _ in range(steps_per_epoch):
                lr = lr_at(gstep, total_steps, float(tcfg["lr"]), float(tcfg["warmup_frac"]),
                           float(tcfg.get("min_lr_ratio", 0.1)))
                for g in optim.param_groups:
                    g["lr"] = lr

                x, y = data_lib.get_batch(bundle.train, batch_size, block_size, device, gen)
                _, loss = model(x, y)
                reg = 0.0
                if l1_lambda:
                    if l1_mode == "mean":
                        reg = l1_lambda * sum(m.weight.abs().mean() for m in qmods) / max(len(qmods), 1)
                    else:
                        reg = l1_lambda * sum(m.weight.abs().sum() for m in qmods)
                total = loss + reg if l1_lambda else loss
                optim.zero_grad(set_to_none=True)
                total.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
                ste = grad_health(qmods) if qmods else float("nan")
                optim.step()
                gstep += 1

                l = loss.item()
                running += l
                running_n += 1
                ep_loss += l
                ep_n += 1

                if gstep % log_iv == 0 or gstep == 1:
                    w_step.writerow([gstep, epoch, f"{lr:.6g}", f"{running / running_n:.4f}",
                                     f"{float(gnorm):.4f}", f"{ste:.4e}", f"{reg:.4f}",
                                     f"{time.time() - t0:.1f}"])
                    running, running_n = 0.0, 0
                    step_csv.flush()

                if qmods and (gstep % diag_iv == 0 or gstep == 1):
                    st = collect_stats(qmods, mcfg.quant["p"], mcfg.quant["skip_zero"])
                    row = {"step": gstep, "epoch": epoch, "val": None, **st.as_dict(), "ste_grad_norm": ste}
                    diag_f.write(json.dumps(row) + "\n")
                    diag_f.flush()
                    hist.append(row)

            val_loss, val_ppl = evaluate(model, data_lib.val_loader(bundle.val, batch_size, block_size, device))
            rec = {"epoch": epoch, "step": gstep, "train_loss_epoch": ep_loss / max(ep_n, 1),
                   "val_loss": val_loss, "val_ppl": val_ppl,
                   "lr": lr, "elapsed_s": round(time.time() - t0, 1)}
            w_ep.writerow([rec["epoch"], rec["step"], f"{rec['train_loss_epoch']:.4f}",
                           f"{val_loss:.4f}", f"{val_ppl:.3f}", f"{lr:.6g}", rec["elapsed_s"]])
            ep_csv.flush()
            print(f"[{tag}] epoch {epoch}/{epochs} | train {rec['train_loss_epoch']:.4f} "
                  f"| val {val_loss:.4f} | ppl {val_ppl:.2f} | lr {lr:.2e} | {rec['elapsed_s']}s")

            if qmods:
                st = collect_stats(qmods, mcfg.quant["p"], mcfg.quant["skip_zero"])
                st.grad_norm = ste
                row = {"step": gstep, "epoch": epoch, "val_ppl": val_ppl, **st.as_dict()}
                diag_f.write(json.dumps(row) + "\n")
                diag_f.flush()

            ckpt = {
                "model": model.state_dict(),
                "config": {"model": asdict(mcfg) | {"quant": mcfg.quant}, "quant": quant},
                "variant": variant,
                "epoch": epoch,
                "step": gstep,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "run": tag,
            }
            torch.save(ckpt, os.path.join(run_dir, "ckpt_last.pt"))
            if val_loss < best["val_loss"]:
                best = {"val_loss": val_loss, "step": gstep, "epoch": epoch}
                torch.save(ckpt, os.path.join(run_dir, "ckpt_best.pt"))
    finally:
        step_csv.close()
        ep_csv.close()
        diag_f.close()

    info["best"] = best
    info["final_val_loss"] = val_loss
    info["final_val_ppl"] = val_ppl
    info["elapsed_s"] = round(time.time() - t0, 1)
    write_json(os.path.join(run_dir, "run.json"), info)
    print(f"[{tag}] готово: val_loss={val_loss:.4f} ppl={val_ppl:.2f} за {info['elapsed_s']}s -> {run_dir}")
    return info


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение модели с QAT (теория шага с пропуском нуля)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variant", choices=["A", "B", "C"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    tag = args.tag or f"{args.variant}_p{cfg['quant']['p']}_" + \
        ("zero" if args.variant == "B" else "skipzero" if args.variant == "C" else "fp32")
    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.seed is not None:
        overrides["seed"] = args.seed
    train_one(cfg, args.variant, tag, overrides)


if __name__ == "__main__":
    main()
