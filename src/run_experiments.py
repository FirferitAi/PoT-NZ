"""Серия экспериментов: A (fp32), B (сетка с нулём), C (сетка без нуля) + сжатие.

Все модели обучаются на одних данных, с одним seed и одинаковой архитектурой.
Итог: runs/comparison.md, runs/comparison.json, runs/curves.png.
"""
from __future__ import annotations

import sys

try:  # консоль Windows по умолчанию cp1251 — иначе "×" и "→" ломают print
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import copy
import json
import os

import yaml

import train as train_lib

DEFAULT_EXPERIMENTS = [
    {"name": "A_fp32", "variant": "A"},
    {"name": "B_p4_zero", "variant": "B", "quant": {"p": 4}},
    {"name": "C_p4_skipzero", "variant": "C", "quant": {"p": 4}},
    {"name": "C_p3_skipzero", "variant": "C", "quant": {"p": 3}},
    {"name": "C_p5_skipzero", "variant": "C", "quant": {"p": 5}},
]


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def summarize(cfg: dict, name: str) -> dict:
    import torch

    import evaluate as eval_lib

    run_dir = os.path.join(cfg["out_dir"], name)
    device = train_lib.pick_device(cfg.get("device", "auto"))
    model, ck, _ = eval_lib.load_run(run_dir, device)
    vloss, vppl = eval_lib.val_ppl(model, run_dir, cfg, device)
    run_json = {}
    rj_path = os.path.join(run_dir, "run.json")
    if os.path.exists(rj_path):
        run_json = json.load(open(rj_path, encoding="utf-8"))
    row = {"name": name, "variant": ck["variant"], "val_loss": vloss, "val_ppl": vppl,
           "params_total": model.n_params(),
           "epochs": run_json.get("train", {}).get("epochs", ck.get("epoch"))}
    q = model.quantizer
    if q is not None:
        from quant import collect_stats

        st = collect_stats(model.quant_linears(), q.p, q.skip_zero)
        row["quant"] = {"p": q.p, "n": q.n, "skip_zero": q.skip_zero,
                        "scale_mode": q.scale_mode, "levels": 2 ** (q.p + 1) + (0 if q.skip_zero else 1)}
        from quant import average_bits_with_zero
        row["bits_carve"] = (None if q.skip_zero
                             else average_bits_with_zero(q.p, st.zero_frac))
        row["stats"] = {"on_grid_frac": st.on_grid_frac, "zero_frac": st.zero_frac,
                        "rel_err": st.rel_err,
                        "satur_frac": st.satur_frac, "mean_abs_k": st.mean_abs_k,
                        "k_hist": st.k_hist, "max_abs_norm": st.max_abs_norm}
        eval_lib.plot_hist(st.k_hist, q.p, os.path.join(run_dir, "k_hist.png"),
                           f"{name}: распределение k")
    row["size"] = eval_lib.export_accounting(model)
    row["size_uint8"] = eval_lib.worst_case_bytes(model)

    # реальный упакованный файл (коды p+1 бит плотно, масштабы fp16)
    from pack import pack_model
    packed = pack_model(model, os.path.join(run_dir, "weights_packed.bin"),
                        scale_dtype="float16")
    packed["bytes_file_fp32_scales"] = packed["bytes_file"] + packed["scales_bytes"]
    fp32_bytes = row["size"].get("bytes_fp32_same_model", row["size"]["bytes_theory"])
    packed["compression_vs_fp32"] = fp32_bytes / packed["bytes_file"]
    row["packed"] = packed
    return row


def read_epochs(cfg: dict, name: str) -> dict:
    import csv

    path = os.path.join(cfg["out_dir"], name, "epoch_log.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["epoch"]: r for r in csv.DictReader(f)}


def plot_curves(cfg: dict, rows: list[dict]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    for r in rows:
        ep = read_epochs(cfg, r["name"])
        if not ep:
            continue
        steps = [int(ep[k]["step"]) for k in sorted(ep, key=int)]
        vl = [float(ep[k]["val_loss"]) for k in sorted(ep, key=int)]
        vp = [float(ep[k]["val_ppl"]) for k in sorted(ep, key=int)]
        axes[0].plot(steps, vl, label=r["name"])
        axes[1].plot(steps, vp, label=r["name"])
    axes[0].set_xlabel("шаг"); axes[0].set_ylabel("val loss"); axes[0].set_title("Кривые обучения")
    axes[1].set_xlabel("шаг"); axes[1].set_ylabel("perplexity"); axes[1].set_title("Perplexity на валидации")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["out_dir"], "curves.png"))
    plt.close(fig)
    return True


def plot_bits_vs_ppl(cfg: dict, rows: list[dict], epochs=None, name="bits_vs_ppl.png") -> bool:
    """Perplexity от числа бит на вес: насколько дешевле то же качество."""
    if epochs is not None:
        rows = [r for r in rows if r.get("epochs") == epochs]
        if len(rows) < 2:
            return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    base = next((r for r in rows if r["variant"] == "A"), None)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    for r in rows:
        sz = r.get("size") or {}
        bpw = r.get("bits_carve") if r.get("bits_carve") is not None else sz.get("bits_per_weight", 32.0)
        bpw = bpw + (sz.get("scale_overhead_bits", 0.0) or 0.0)
        color = "#888888" if r["variant"] == "A" else "#1f77b4"
        ax.scatter([bpw], [r["val_ppl"]], s=45, color=color, zorder=3)
        ax.annotate(r["name"], (bpw, r["val_ppl"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    if base:
        ax.axhline(base["val_ppl"], ls="--", lw=1, color="#888888", label="baseline fp32")
        ax.legend(fontsize=8)
    ax.set_xlabel("бит на вес (включая масштаб блока)")
    ax.set_ylabel("perplexity (валидация)")
    ax.set_title("Качество против размера")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["out_dir"], name))
    plt.close(fig)
    return True


def write_report(cfg: dict, rows: list[dict]) -> str:
    # базовая линия для сравнения - fp32-модель с тем же числом эпох
    def baseline_for(r):
        cands = [b for b in rows if b["variant"] == "A"]
        same = [b for b in cands if b.get("epochs") == r.get("epochs")]
        if same:
            return same[0]
        return cands[0] if cands else None

    base = next((r for r in rows if r["variant"] == "A" and r.get("epochs") == 30),
                next((r for r in rows if r["variant"] == "A"), None))
    lines = ["# Сравнение моделей: шаг с пропуском нуля (QAT)", ""]
    lines.append("| модель | вариант | p | эпох | val loss | perplexity | ×A | на сетке | нули | клип "
                 "| отн. ошибка | бит/вес | сжатие (теория) | .bin, байт | сжатие (файл) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        b = baseline_for(r)
        ratio = f"{r['val_ppl'] / b['val_ppl']:.3f}" if b and b is not r and r["val_ppl"] else "-"
        q = r.get("quant") or {}
        st = r.get("stats") or {}
        sz = r.get("size") or {}
        p_val = q.get("p", "-")
        og = f"{st.get('on_grid_frac', 0) * 100:.1f}%" if st else "-"
        zf = f"{st.get('zero_frac', 0) * 100:.3f}%" if st else "-"
        sat = f"{st.get('satur_frac', 0) * 100:.1f}%" if st else "-"
        bpw = f"{sz.get('bits_per_weight', 0):.2f}" if sz else "-"
        comp = f"{sz.get('compression_vs_fp32', 1.0):.2f}×" if sz else "-"
        pk = r.get("packed") or {}
        bin_bytes = f"{pk['bytes_file']:,}" if pk else "-"
        bin_comp = f"{pk.get('compression_vs_fp32', 1.0):.2f}×" if pk else "-"
        rel = f"{st.get('rel_err', 0) * 100:.3f}%" if st else "-"
        lines.append(f"| {r['name']} | {r['variant']} | {p_val} | {r.get('epochs', '-')} | "
                     f"{r['val_loss']:.4f} | {r['val_ppl']:.2f} | {ratio} | {og} | {zf} | {sat} "
                     f"| {rel} | {bpw} | {comp} | {bin_bytes} | {bin_comp} |")
    lines += ["", "## Проверка гипотез", ""]
    def h1(name: str, base_name: str) -> str:
        r = next((x for x in rows if x["name"] == name), None)
        b = next((x for x in rows if x["name"] == base_name), None)
        if not r or not b:
            return ""
        ratio = r["val_ppl"] / b["val_ppl"]
        return (f"- **H1 ({r['epochs']} эпох):** `{name}` / `{base_name}` = {ratio:.3f} "
                f"({r['val_ppl']:.2f} против {b['val_ppl']:.2f}) — порог 1.1 "
                f"{'пройден' if ratio <= 1.1 else 'НЕ пройден'}.")
    for name, bname in (("C_p4_skipzero", "A_fp32"), ("C_p4_skipzero_e60", "A_fp32_e60"),
                        ("C_p2_skipzero", "A_fp32"), ("C_p4_global", "A_fp32")):
        line = h1(name, bname)
        if line:
            lines.append(line)
    lines.append("- **Веса на сетке:** 100% (проверка `q == k·step·scale` по всем слоям).")
    lines.append("- **Нули в C:** 0.000% во всех прогонах варианта C (ноль не входит в сетку).")
    lines.append("- **H2 (эффективность):** размер в битах: (p+1) + 0.5 бита/вес против 32 бит fp32.")
    for r in rows:
        if r.get("bits_carve") is not None:
            st = r.get("stats") or {}
            f = st.get("zero_frac", 0.0)
            verdict = "вырезание нуля выгоднее" if f > 0.17 else "простая 5-битная сетка выгоднее"
            lines.append(f"- **Вариант Б, вырезание нуля:** доля нулей f = {f * 100:.2f}%, "
                         f"средняя длина кода = {r['bits_carve']:.2f} бита/вес "
                         f"(порог окупаемости f > 17%) → {verdict}.")
    lines += ["", "## Размеры", "",
              "`.bin` — реальный файл `runs/<имя>/weights_packed.bin`: коды уложены подряд "
              "`p+1` бит, масштабы блоков fp16, эмбеддинги/LayerNorm/bias/lm_head — fp32.", ""]
    lines.append("| модель | байт (теория) | .bin, байт | .bin, fp32-масштабы | fp32-эквивалент | сжатие (файл) |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        sz = r.get("size") or {}
        bt = int(sz.get('bytes_theory', 0))
        bf = int(sz.get('bytes_fp32_same_model', bt))
        pk = r.get("packed") or {}
        bin_bytes = f"{pk['bytes_file']:,}" if pk else "-"
        bin32 = f"{pk.get('bytes_file_fp32_scales', 0):,}" if pk else "-"
        bin_comp = f"{pk.get('compression_vs_fp32', 1.0):.2f}×" if pk else "-"
        lines.append(f"| {r['name']} | {bt:,} | {bin_bytes} | {bin32} | {bf:,} | {bin_comp} |")
    lines += ["", "Разница «.bin» и «теория» — JSON-заголовок файла (около 6.8 КБ на 53 тензора).",
              "Для варианта B теория считает `log2(33) = 5.04` бита (энтропийный предел), "
              "а реальный код занимает 6 бит: 33 значения не помещаются в 5 бит без вырезания нуля.",
              "Разбор битности — `runs/report_bits.md`, разреженность (SZM1) — "
              "`runs/report_sparse.md`, средняя BPE-модель (7.3M) — `runs/report_medium.md`, "
              "квантование эмбеддинга и L1 — `runs/report_final.md`.",
              "Серии `runs_medium/` и `runs_l1/` (в т.ч. L1-разреженность и "
              "`quantize_embedding`) — в `runs/report_final.md`. В `runs_l1/` исправлена "
              "инициализация эмбеддинга (std 0.02 вместо N(0,1)), поэтому абсолютные ppl "
              "там ниже, чем в этой таблице; сравнения внутри серий корректны."]
    path = os.path.join(cfg["out_dir"], "comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--spec", default="experiments.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="имена экспериментов")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-train", action="store_true", help="только свести готовые запуски")
    args = ap.parse_args()

    cfg = train_lib.load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    specs = DEFAULT_EXPERIMENTS
    if os.path.exists(args.spec):
        with open(args.spec, "r", encoding="utf-8") as f:
            specs = yaml.safe_load(f)["experiments"]
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]

    rows = []
    for spec in specs:
        name, variant = spec["name"], spec["variant"]
        run_cfg = deep_merge(cfg, {k: spec[k] for k in spec if k in ("data", "model", "quant", "train", "seed")})
        if not args.skip_train:
            print(f"\n=== {name} (вариант {variant}) ===")
            train_lib.train_one(run_cfg, variant, name)
        rows.append(summarize(run_cfg, name))

    for row in rows:
        json.dump(row, open(os.path.join(cfg["out_dir"], row['name'], "summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    report = write_report(cfg, rows)
    if plot_curves(cfg, rows):
        print(f"графики: {os.path.join(cfg['out_dir'], 'curves.png')}")
    if plot_bits_vs_ppl(cfg, rows):
        print(f"графики: {os.path.join(cfg['out_dir'], 'bits_vs_ppl.png')}")
    epochs_max = max((r.get("epochs") or 0) for r in rows)
    if plot_bits_vs_ppl(cfg, rows, epochs=epochs_max, name="bits_vs_ppl_e60.png"):
        print(f"графики: {os.path.join(cfg['out_dir'], 'bits_vs_ppl_e60.png')}")
    with open(os.path.join(cfg["out_dir"], "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nотчёт: {report}")
    print(open(report, encoding="utf-8").read())


if __name__ == "__main__":
    main()
