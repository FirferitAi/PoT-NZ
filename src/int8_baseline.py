"""Сравнение теории с обычным int8: PTQ-квантование обученной fp32-модели.

int8 (симметричный, стандарт):
    scale = absmax / 127,  q = clamp(round(w/scale), -127, 127) * scale
    в сетке есть ноль, кодов 255 из 256 (ноль один), сторона ±127.
Теория (p=7, 8 бит):
    сетка k·n/2^p, k ∈ {-128..-1, 1..128} — 256 значений, ноль отсутствует,
    сторона ±128, шаг кратен степени двойки, zero-point не нужен.

Здесь int8 применяется к уже обученной fp32-модели (PTQ, без дообучения) —
это нижняя оценка качества int8, верхнюю даёт int8-QAT.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

import train as train_lib
from evaluate import load_run, val_ppl


def quant_int8_block(w: torch.Tensor, block: int = 32) -> torch.Tensor:
    flat = w.reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    scale = blocks.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.clamp(torch.round(blocks / scale[:, None]), -127, 127) * scale[:, None]
    return q.reshape(-1)[: w.numel()].view_as(w)


def quant_int8_tensor(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().max().clamp_min(1e-12) / 127.0
    return torch.clamp(torch.round(w / scale), -127, 127) * scale


def int8_bytes(numel: int, block: int, scale_bytes: int = 2) -> int:
    return numel + math.ceil(numel / block) * scale_bytes


def quantize_model(model, mode: str, block: int = 32) -> int:
    """Квантует линейные слои внутри blocks.* (attention/FFN), lm_head не трогаем."""
    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not name.startswith("blocks."):
                continue
            if isinstance(mod, torch.nn.Linear):
                q = (quant_int8_block(mod.weight, block) if mode == "block"
                     else quant_int8_tensor(mod.weight))
                mod.weight.copy_(q)
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/A_fp32_e60", help="fp32-модель для PTQ")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    cfg = train_lib.load_config(args.config)
    device = torch.device(args.device)
    out = {"source_run": args.run, "baseline_fp32_ppl": None, "int8": {}}

    model, ck, _ = load_run(args.run, device)
    base_loss, base_ppl = val_ppl(model, args.run, cfg, device)
    out["baseline_fp32_ppl"] = base_ppl
    print(f"fp32 baseline: ppl {base_ppl:.3f}")

    n_q = sum(1 for n, m in model.named_modules()
              if n.startswith("blocks.") and isinstance(m, torch.nn.Linear))
    numel = sum(m.weight.numel() for n, m in model.named_modules()
                if n.startswith("blocks.") and isinstance(m, torch.nn.Linear))
    other = sum(p.numel() for p in model.parameters()) - numel

    for mode in ("block", "tensor"):
        m2, _, _ = load_run(args.run, device)
        used = quantize_model(m2, mode, args.block)
        _, ppl = val_ppl(m2, args.run, cfg, device)
        sb = 2 if mode == "block" else 4
        size = int8_bytes(numel, args.block if mode == "block" else numel, sb) + other * 4
        out["int8"][mode] = {
            "layers": used, "val_ppl": ppl, "ratio_vs_fp32": ppl / base_ppl,
            "bits_per_weight": 8.0 + (sb * 8 / (args.block if mode == "block" else numel)),
            "bytes_file": size, "compression_vs_fp32": (numel + other) * 4 / size,
        }
        print(f"int8 {mode:6s}: ppl {ppl:.3f} ({ppl / base_ppl:.3f}× fp32), "
              f"бит/вес {out['int8'][mode]['bits_per_weight']:.2f}, "
              f"файл {size:,} байт ({out['int8'][mode]['compression_vs_fp32']:.2f}×)")

    # наша модель той же разрядности (p=7 -> 8 бит кода), если обучена
    p7 = "runs/C_p7_skipzero_e60"
    if os.path.exists(os.path.join(p7, "summary.json")):
        sm = json.load(open(os.path.join(p7, "summary.json"), encoding="utf-8"))
        out["skip_zero_p7"] = {
            "val_ppl": sm["val_ppl"],
            "bits_per_weight": sm["size"]["bits_per_weight"] + sm["size"]["scale_overhead_bits"],
            "bytes_file": sm.get("packed", {}).get("bytes_file"),
            "compression_vs_fp32": sm.get("packed", {}).get("compression_vs_fp32"),
        }
        print(f"теория p=7 : ppl {sm['val_ppl']:.3f}, "
              f"бит/вес {out['skip_zero_p7']['bits_per_weight']:.2f}, "
              f"файл {out['skip_zero_p7']['bytes_file']:,} байт")

    path = os.path.join(os.path.dirname(args.run), "int8_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
