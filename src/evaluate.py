"""Оценка обученной модели: perplexity, диагностика квантования, размер в битах."""
from __future__ import annotations

import sys

try:  # консоль Windows по умолчанию cp1251 — иначе "×" и "→" ломают print
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import json
import math
import os

import torch

import data as data_lib
from model import ModelConfig, TinyGPT
from pack import block_scales, pack_model
from quant import collect_stats, encode_k, encode_k_with_zero, bits_per_weight


def load_run(run_dir: str, device: torch.device, ckpt: str = "ckpt_best.pt"):
    path = os.path.join(run_dir, ckpt)
    if not os.path.exists(path):
        path = os.path.join(run_dir, "ckpt_last.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ck["config"]["model"])
    model = TinyGPT(mcfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = data_lib.load_tokenizer(run_dir)
    return model, ck, tok


def val_ppl(model: TinyGPT, run_dir: str, cfg: dict, device: torch.device) -> tuple[float, float]:
    text = data_lib.load_text(cfg["data"]["path"], cfg["data"].get("url"))
    n_val = int(len(text) * float(cfg["data"]["val_fraction"]))
    tok = data_lib.load_tokenizer(run_dir)
    val = tok.encode(text[len(text) - n_val:])
    bs = int(cfg["train"]["batch_size"])
    loader = data_lib.val_loader(val, bs, int(cfg["data"]["block_size"]), device)
    total, ntok = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            _, loss = model(x, y)
            total += loss.item() * y.numel()
            ntok += y.numel()
    mean = total / max(ntok, 1)
    return mean, math.exp(mean)


def export_accounting(model: TinyGPT) -> dict:
    """Теоретический размер весов (план, раздел 12 / теория, разделы 4, 11)."""
    q = model.quantizer
    params = sum(p.numel() for p in model.parameters())
    if q is None:
        return {"mode": "fp32", "weights": params, "bits_per_weight": 32.0,
                "bytes_theory": params * 4, "scale_overhead_bits": 0.0}

    n_q = model.n_quant_params()
    total = 0.0
    for m in model.quant_linears():
        numel = m.weight.numel()
        bits = bits_per_weight(q.p, q.skip_zero)
        codes = math.ceil(numel * bits / 8)
        scales = 0 if q.scale_mode == "global" else math.ceil(numel / q.block_size) * 2
        total += codes + scales
    fp32_other = (params - n_q) * 4
    return {
        "mode": "skip_zero" if q.skip_zero else "with_zero",
        "weights": params,
        "quant_weights": n_q,
        "bits_per_weight": bits_per_weight(q.p, q.skip_zero),
        "scale_overhead_bits": 0.0 if q.scale_mode == "global" else 16.0 / q.block_size,
        "bytes_theory": total + fp32_other,
        "bytes_fp32_same_model": params * 4,
        "compression_vs_fp32": (params * 4) / max(total + fp32_other, 1),
    }


def worst_case_bytes(model: TinyGPT) -> dict:
    """Реальный файл: коды в uint8 + scale fp16 без упаковки бит (план: упаковку не делаем)."""
    q = model.quantizer
    if q is None:
        return {"bytes_file": sum(p.numel() * 4 for p in model.parameters()), "packed": False}
    n_q = model.n_quant_params()
    codes_bytes = n_q  # 1 байт на код: запас без битовой упаковки
    scales_bytes = 0 if q.scale_mode == "global" else math.ceil(n_q / q.block_size) * 2
    other = (sum(p.numel() for p in model.parameters()) - n_q) * 4
    return {"bytes_file": codes_bytes + scales_bytes + other, "packed": False}


@torch.no_grad()
def save_deploy(model: TinyGPT, run_dir: str) -> dict | None:
    q = model.quantizer
    if q is None:
        return None
    arrays = {}
    meta = {"p": q.p, "n": q.n, "skip_zero": q.skip_zero,
            "scale_mode": q.scale_mode, "block_size": q.block_size,
            "codec": "skip_zero(p+1 bit)" if q.skip_zero else "with_zero(2^(p+1)+1 levels)",
            "scale_dtype": "float32 (для побитовой проверки; в формате хранения — fp16, 2 байта/блок)",
            "layers": []}
    for i, m in enumerate(model.quant_linears()):
        _, k, _, scale = m.quantizer.quantize_details(m.weight.detach())
        if q.skip_zero:
            codes = encode_k(k, q.p)
            assert int((k == 0).sum()) == 0, "в модели C найден ноль"
        else:
            codes = encode_k_with_zero(k, q.p)
        codes = codes.to(torch.uint8).cpu().numpy()
        blocks = block_scales(m.weight.detach(), q)
        arrays[f"codes_{i}"] = codes.reshape(-1)
        if q.scale_mode != "global":
            arrays[f"scale_{i}"] = blocks
        meta["layers"].append({"name": f"linear_{i}", "shape": list(m.weight.shape),
                               "kmin": int(k.min()), "kmax": int(k.max()),
                               "scales": int(blocks.size)})
    import numpy as np
    path = os.path.join(run_dir, "deploy_codes.npz")
    np.savez_compressed(path, **arrays)
    with open(os.path.join(run_dir, "deploy_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {"file": path, "bytes_file": os.path.getsize(path)}


@torch.no_grad()
def sample_text(model: TinyGPT, tok, device, prompt: str, n_new: int = 300, temp: float = 0.8) -> str:
    ids = tok.encode(prompt)
    x = ids.to(device).unsqueeze(0)
    out = list(ids)
    for _ in range(n_new):
        ctx = x[:, -model.cfg.context:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :] / max(temp, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        out.append(int(nxt.item()))
        x = torch.cat([x, nxt], dim=1)
    return tok.decode(out)


def plot_hist(stats_hist: list[int], p: int, path: str, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ks = list(range(-2 ** p, 2 ** p + 1))
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=140)
    ax.bar(ks, stats_hist, width=0.85)
    ax.set_xlabel("k (вес = k * n / 2^p * scale)")
    ax.set_ylabel("число весов")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="каталог запуска, напр. runs/C_p4_skipzero")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()

    import train as train_lib

    cfg = train_lib.load_config(args.config)
    device = train_lib.pick_device(args.device)
    model, ck, tok = load_run(args.run, device)
    q = model.quantizer

    vloss, vppl = val_ppl(model, args.run, cfg, device)
    result = {
        "run": os.path.basename(args.run.rstrip("/\\")),
        "variant": ck["variant"],
        "checkpoint_epoch": ck["epoch"],
        "val_loss": vloss,
        "val_ppl": vppl,
        "params_total": model.n_params(),
        "params_quant": model.n_quant_params(),
    }
    if q is not None:
        st = collect_stats(model.quant_linears(), q.p, q.skip_zero)
        result.update({"quant": {"p": q.p, "n": q.n, "skip_zero": q.skip_zero,
                                "scale_mode": q.scale_mode, "block_size": q.block_size,
                                "step": q.step, "levels": (2 ** (q.p + 1)) if q.skip_zero else 2 ** (q.p + 1) + 1},
                       "stats": st.as_dict()})
        hist_path = os.path.join(args.run, "k_hist.png")
        result["hist_plot"] = plot_hist(st.k_hist, q.p, hist_path, f"{result['run']} распределение k")
        dep = save_deploy(model, args.run)
        if dep:
            result["deploy"] = dep
            packed = pack_model(model, os.path.join(args.run, "weights_packed.bin"),
                                scale_dtype="float16")
            packed["bytes_file_fp32_scales"] = packed["bytes_file"] + packed["scales_bytes"]
            result["packed"] = packed
    result["size_theory"] = export_accounting(model)
    result["size_uint8_file"] = worst_case_bytes(model)

    if args.sample:
        prompt = "ROMEO:"
        result["sample"] = sample_text(model, tok, device, prompt)

    out = os.path.join(args.run, "eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("stats", "sample")},
                     ensure_ascii=False, indent=2))
    if "stats" in result:
        s = result["stats"]
        print(f"\nна сетке: {s['on_grid_frac']*100:.2f}% | нули: {s['zero_frac']*100:.3f}% "
              f"| клип: {s['satur_frac']*100:.2f}% | mean|k| {s['mean_abs_k']:.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
