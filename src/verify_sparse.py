"""Проверка упаковки с битовой маской нулей (формат SZM1).

Запуск:
    python src/verify_sparse.py runs/B_p4_zero [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

import evaluate as eval_lib
import train as train_lib
from pack import (
    MAGIC_SPARSE,
    pack_model,
    pack_sparse,
    read_header,
    unpack_sparse,
)
from quant import QuantLinear


def sparse_bytes(numel: int, f: float, bits: int, block: int, scale_bytes: int) -> int:
    """Размер разреженной части (без заголовка/fp32) при доле нулей f."""
    nonzero = int(round(numel * (1.0 - f)))
    return math.ceil(numel / 8) + math.ceil(nonzero * bits / 8) + math.ceil(numel / block) * scale_bytes


def dense_bytes(numel: int, bits: int, block: int, scale_bytes: int) -> int:
    return math.ceil(numel * bits / 8) + math.ceil(numel / block) * scale_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/B_p4_zero")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = args.run.rstrip("/\\")
    cfg = train_lib.load_config(args.config)
    device = torch.device(args.device)
    print(f"== {run_dir} ==")

    model, ck, _ = eval_lib.load_run(run_dir, device)
    q = model.quantizer
    if q is None:
        print("модель fp32 — разреженная упаковка не применима")
        return

    dense_path = os.path.join(run_dir, "weights_packed.bin")
    sparse_path = os.path.join(run_dir, "weights_sparse.bin")
    dense = pack_model(model, dense_path, scale_dtype="float16")
    sparse = pack_sparse(model, sparse_path, scale_dtype="float16")

    # побитовая проверка при масштабах fp32
    sparse32_path = os.path.join(run_dir, "weights_sparse_fp32scale.bin")
    pack_sparse(model, sparse32_path, scale_dtype="float32")
    _, st16 = unpack_sparse(sparse_path)
    _, st32 = unpack_sparse(sparse32_path)

    n_zero_ok, n_layers, max_err16, exact32, zeros_total = 0, 0, 0.0, 0, 0
    for name, mod in model.named_modules():
        if not isinstance(mod, QuantLinear):
            continue
        ref, k, _, _ = mod.quantizer.quantize_details(mod.weight.detach())
        ref = ref.detach().cpu()
        key = f"{name}.weight"
        got16, got32 = st16[key], st32[key]
        n_layers += 1
        exact32 += int(torch.equal(got32, ref))
        zeros_total += int((k == 0).sum())
        if int((k == 0).sum()):
            n_zero_ok += int(torch.equal(got32[k.cpu() == 0], torch.zeros_like(got32[k.cpu() == 0])))
        max_err16 = max(max_err16, (got16 - ref).abs().max().item())

    header = read_header(sparse_path)
    print(f"формат: {MAGIC_SPARSE.decode()}, p={q.p}, бит на ненулевой код: {q.p + 1}")
    print(f"разреженность: {sparse['zero']:,} нулей из {sparse['zero'] + sparse['nonzero']:,} "
          f"весов ({sparse['zero_frac'] * 100:.2f}%)")
    print(f"слоёв: {n_layers}, побитово восстановлено (fp32-масштабы): {exact32}/{n_layers}")
    print(f"нулей на своих местах: {n_zero_ok}/{n_layers} слоёв (всего {zeros_total:,})")
    print(f"расхождение при fp16-масштабах: {max_err16:g}")
    print(f"ppl заголовка: {ck['val_ppl']:.3f}")

    # качество восстановленной модели
    mcfg = type(model.cfg)(**vars(model.cfg))
    recon = type(model)(mcfg).to(device)
    recon.load_state_dict(st16)
    recon.eval()
    _, ppl = eval_lib.val_ppl(recon, run_dir, cfg, device)
    print(f"perplexity из разреженного .bin: {ppl:.3f} (исходная {ck['val_ppl']:.3f})")

    print("\nразмер:")
    for label, size in (("плотный .bin (6 бит на все веса)", dense["bytes_file"]),
                        ("разреженный .bin (1 бит маски + 5 бит)", sparse["bytes_file"])):
        print(f"  {label:45s} {size:>10,} байт")
    print(f"  выигрыш разреженного: {dense['bytes_file'] / sparse['bytes_file']:.3f}× "
          f"({(1 - sparse['bytes_file'] / dense['bytes_file']) * 100:.2f}%)")
    print(f"  из чего: маска {sparse['mask_bytes']:,} + коды {sparse['codes_bytes']:,} "
          f"+ масштабы {sparse['scales_bytes']:,} + fp32 {sparse['fp32_bytes']:,} "
          f"+ заголовок {sparse['header_bytes']:,}")

    # точка окупаемости: разреженный vs плотный 6-битный и vs 5-битный skip-zero
    # разреженный: 1 бит маски + (p+1)*(1-f) бит кодов на вес
    # плотный X:   X бит на вес  ->  равенство при f = 1 - (X-1)/(p+1)
    numel = sparse["nonzero"] + sparse["zero"]
    block = q.block_size
    print("порог окупаемости (по формату, без заголовка и fp32):")
    print(f"  1 бит маски + {q.p + 1}·(1-f) бит кодов против плотных X бит: f > 1 - (X-1)/{q.p + 1}")
    print(f"  фактическая доля нулей: {sparse['zero_frac'] * 100:.2f}%")
    break_even = {}
    for label, X in (("плотный B (6 бит на код)", 6), ("плотный C (5 бит на код)", 5)):
        f_be = max(0.0, 1 - (X - 1) / (q.p + 1))
        break_even[label] = f_be
        verdict = "разреженный меньше при любой доле нулей" if f_be == 0 else (
            "разреженный выигрывает" if sparse["zero_frac"] > f_be else "разреженный проигрывает")
        print(f"  vs {label}: f > {f_be * 100:.1f}% — {verdict}")

    table = []
    for f in (0.0, 0.02, 0.0583, 0.10, 0.15, 0.20, 0.30, 0.40):
        s = sparse_bytes(numel, f, q.p + 1, block, 2)
        d = dense_bytes(numel, 6, block, 2)
        c = dense_bytes(numel, 5, block, 2)
        table.append({"zero_frac": f, "sparse": s, "dense_b6": d, "dense_c5": c})
    print("\n  f (нули) | разреженный | плотный B (6 бит) | плотный C (5 бит)")
    for row in table:
        print(f"  {row['zero_frac'] * 100:6.2f}%  | {row['sparse']:>10,} | "
              f"{row['dense_b6']:>16,} | {row['dense_c5']:>15,}")

    out = {
        "run": os.path.basename(run_dir),
        "p": q.p,
        "zero_frac": sparse["zero_frac"],
        "dense_bytes": dense["bytes_file"],
        "sparse_bytes": sparse["bytes_file"],
        "sparse_gain": dense["bytes_file"] / sparse["bytes_file"],
        "exact_layers_fp32scale": f"{exact32}/{n_layers}",
        "ppl_from_sparse": ppl,
        "ppl_original": ck["val_ppl"],
        "break_even": break_even,
        "table": table,
    }
    path = os.path.join(os.path.dirname(run_dir) or ".", "sparse_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
