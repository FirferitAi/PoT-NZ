"""Диапазон весов эмбеддинга/головы до и после QAT (эксперимент с quantize_embedding).

Запуск:
    python src/check_embedding.py --config config_medium.yaml \
        --runs A_fp32_medium C_p7_medium C_p7E_medium
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

import train as train_lib
from evaluate import load_run
from model import QuantEmbedding, QuantHead
from pack import pack_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_medium.yaml")
    ap.add_argument("--runs", nargs="+", default=["A_fp32_medium", "C_p7_medium", "C_p7E_medium"])
    args = ap.parse_args()

    cfg = train_lib.load_config(args.config)
    out_dir = cfg["out_dir"]
    device = torch.device("cpu")
    rows = []

    for name in args.runs:
        run_dir = os.path.join(out_dir, name)
        model, ck, _ = load_run(run_dir, device)
        emb_mod = model.tok_emb
        emb = emb_mod.weight.detach()
        quantized = isinstance(emb_mod, QuantEmbedding)
        row = {
            "run": name,
            "embedding_quantized": quantized,
            "head": type(model.lm_head).__name__,
            "raw_absmax": emb.abs().max().item(),
            "raw_mean_abs": emb.abs().mean().item(),
            "frac_gt1": (emb.abs() > 1).float().mean().item(),
            "params": model.n_params(),
            "quant_params": model.n_quant_params(),
        }
        if quantized:
            q = emb_mod.quantizer
            qq, k, u, _ = q.quantize_details(emb)
            row.update({
                "quant_absmax": qq.abs().max().item(),
                "max_abs_u_over_n": (u.abs().max() / q.n).item(),
                "clip_frac": (k.abs() == q.max_k).float().mean().item(),
                "zero_frac_emb": (k == 0).float().mean().item(),
                "step": q.step,
            })
        packed = pack_model(model, os.path.join(run_dir, "weights_packed.bin"), scale_dtype="float16")
        row["bin_bytes"] = packed["bytes_file"]
        row["fp32_bytes"] = sum(p.numel() for p in model.parameters()) * 4
        row["compression"] = row["fp32_bytes"] / packed["bytes_file"]
        rj = json.load(open(os.path.join(run_dir, "run.json"), encoding="utf-8"))
        row["ppl_best"] = math.exp(rj["best"]["val_loss"])
        row["ppl_final"] = rj["final_val_ppl"]
        rows.append(row)
        print(f"{name}: emb {'QAT' if quantized else 'fp32'}, raw absmax {row['raw_absmax']:.4f}, "
              f".bin {packed['bytes_file']:,} ({row['compression']:.2f}×), "
              f"ppl best {row['ppl_best']:.1f}")
        if quantized:
            print(f"    quant absmax {row['quant_absmax']:.4f}, max|u|/n {row['max_abs_u_over_n']:.4f}, "
                  f"клип {row['clip_frac'] * 100:.2f}%, нули в эмбеддинге {row['zero_frac_emb'] * 100:.3f}%")

    path = os.path.join(out_dir, "embedding_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
