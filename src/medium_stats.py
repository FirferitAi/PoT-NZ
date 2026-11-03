"""Статистика средней BPE-модели: токенизация, диапазоны весов, размеры .bin.

Запуск:
    python src/medium_stats.py --config config_medium.yaml --runs A_fp32_medium C_p4_medium
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import data as data_lib
import train as train_lib
from evaluate import export_accounting, load_run, val_ppl
from pack import pack_model
from quant import StepQuantizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_medium.yaml")
    ap.add_argument("--runs", nargs="+", default=["A_fp32_medium", "C_p4_medium", "C_p5_medium"])
    args = ap.parse_args()

    cfg = train_lib.load_config(args.config)
    out_dir = cfg["out_dir"]
    device = torch.device("cpu")

    bundle = data_lib.make_bundle({**cfg["data"], "block_size": cfg["data"]["block_size"]})
    tok = bundle.tok
    text = data_lib.load_text(cfg["data"]["path"], cfg["data"].get("url"))
    n_tok = len(bundle.train) + len(bundle.val)
    chars_per_token = len(text) / n_tok
    train_set = set(bundle.train.tolist())
    val_ids = bundle.val.tolist()
    unseen = sum(1 for t in val_ids if t not in train_set)
    stats = {
        "vocab": tok.vocab_size,
        "tokens_total": n_tok,
        "chars_per_token": chars_per_token,
        "val_tokens": len(val_ids),
        "val_tokens_unseen": unseen,
        "val_unseen_frac": unseen / max(len(val_ids), 1),
        "train_tokens": len(bundle.train),
        "occurrences_per_vocab": len(bundle.train) / tok.vocab_size,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # диапазоны весов на fp32-модели
    model, ck, _ = load_run(os.path.join(out_dir, args.runs[0]), device)
    emb = model.tok_emb.weight.detach()
    print("\nэмбеддинг: absmax %.3f, mean|w| %.4f, доля |w|>1: %.4f%%"
          % (emb.abs().max(), emb.abs().mean(), (emb.abs() > 1).float().mean() * 100))
    # что было бы, если квантовать эмбеддинг глобальной сеткой n=1 (как в первых прогонах)
    q = StepQuantizer(n=1.0, p=4, skip_zero=True, scale_mode="global")
    with torch.no_grad():
        k = q._k_from_u(emb)
        clipped = (k.abs() == q.max_k).float().mean().item() * 100
        print(f"если квантовать эмбеддинг глобально (n=1, p=4): клип {clipped:.2f}% весов")
    with torch.no_grad():
        qb = StepQuantizer(n=1.0, p=4, skip_zero=True, scale_mode="per_block", block_size=32)
        u = emb / qb.compute_scale(emb)
        print(f"per-block: |u|/n max = {(u.abs().max() / qb.n):.4f} (масштаб поглощает диапазон)")

    rows = []
    for name in args.runs:
        run_dir = os.path.join(out_dir, name)
        m, ck, _ = load_run(run_dir, device)
        rc = m.n_quant_params()
        acct = export_accounting(m)
        packed = pack_model(m, os.path.join(run_dir, "weights_packed.bin"), scale_dtype="float16")
        fp32_bytes = acct.get("bytes_fp32_same_model", acct["bytes_theory"])
        packed["compression_vs_fp32"] = fp32_bytes / packed["bytes_file"]
        rows.append({
            "run": name,
            "params": m.n_params(),
            "params_quant": rc,
            "fp32_share": 1 - rc / m.n_params(),
            "best_val_ppl": json.load(open(os.path.join(run_dir, "run.json"), encoding="utf-8"))["best"],
            "packed": packed,
            "size": acct,
        })
        print(f"\n{name}: params {m.n_params():,} (квантуется {rc:,}, "
              f"fp32-доля {(1 - rc / m.n_params()) * 100:.1f}%)")
        print(f"  .bin {packed['bytes_file']:,} байт, сжатие {packed['compression_vs_fp32']:.2f}×")

    # сколько было бы при квантовании эмбеддингов/головы (аналитически)
    m = load_run(os.path.join(out_dir, args.runs[0]), device)[0]
    total = m.n_params()
    qn = m.n_quant_params()
    allq = total * 5.5 / 8
    print(f"\nесли квантовать ВСЕ параметры (5.5 бита): {allq:,.0f} байт "
          f"({total * 4 / allq:.2f}× против fp32)")

    path = os.path.join(out_dir, "medium_stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"bpe": stats, "runs": rows,
                   "all_quantized_bytes": allq}, f, ensure_ascii=False, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
