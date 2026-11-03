"""Сводка по L1-эксперименту: доля нулей, ppl, размеры плотного и SZM1-файла.

Запуск:
    python src/l1_report.py --config config_l1.yaml \
        --runs B_p7_l1_0 B_p7_l1_1e5 ... C_p4_init2
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

import train as train_lib
from evaluate import load_run
from pack import pack_model, pack_sparse
from quant import collect_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_l1.yaml")
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    cfg = train_lib.load_config(args.config)
    out_dir = cfg["out_dir"]
    device = torch.device("cpu")
    rows = []

    for name in args.runs:
        run_dir = os.path.join(out_dir, name)
        model, ck, _ = load_run(run_dir, device)
        q = model.quantizer
        rj = json.load(open(os.path.join(run_dir, "run.json"), encoding="utf-8"))
        row = {"run": name, "variant": ck["variant"], "p": None if q is None else q.p,
               "l1_lambda": rj["train"].get("l1_lambda", 0.0),
               "ppl_final": rj["final_val_ppl"], "ppl_best": math.exp(rj["best"]["val_loss"]),
               "params": model.n_params()}
        dense = pack_model(model, os.path.join(run_dir, "weights_packed.bin"), scale_dtype="float16")
        row["dense_bytes"] = dense["bytes_file"]
        if q is not None:
            st = collect_stats(model.quant_linears(), q.p, q.skip_zero)
            row["zero_frac"] = st.zero_frac
            row["on_grid"] = st.on_grid_frac
            row["rel_err"] = st.rel_err
            if q.skip_zero:
                row["sparse_bytes"] = None
                row["bits_per_weight"] = q.p + 1 + 16.0 / q.block_size
            else:
                sp = pack_sparse(model, os.path.join(run_dir, "weights_sparse.bin"),
                                 scale_dtype="float16")
                row["sparse_bytes"] = sp["bytes_file"]
                code_bits = 1 + (q.p + 1) * (1.0 - st.zero_frac)
                row["bits_per_weight"] = code_bits + 16.0 / q.block_size
        rows.append(row)
        b = row.get("bits_per_weight")
        extra = (f", нули {row['zero_frac'] * 100:5.2f}%, SZM1 {row['sparse_bytes']:,} б"
                 if row.get("sparse_bytes") else "")
        print(f"{name:16s} p={row['p']} λ={row['l1_lambda']:g}  ppl {row['ppl_final']:5.2f}  "
              f"плотный {row['dense_bytes']:,} б{extra}  {b:.2f} бит/вес" if b else name)

    path = os.path.join(out_dir, "l1_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
