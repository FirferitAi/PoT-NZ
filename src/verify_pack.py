"""Проверка упаковщика: unpack(pack(codes)) == codes без потерь + реальный размер .bin.

Запуск:
    python src/verify_pack.py runs/C_p4_skipzero [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

import evaluate as eval_lib
import train as train_lib
from model import ModelConfig, TinyGPT
from pack import bits_for_codes, pack_model, pack_weights, unpack_model, unpack_weights
from quant import QuantLinear


def verify_codes(run_dir: str) -> dict:
    """Побитовая проверка упаковки кодов из deploy_codes.npz."""
    npz = os.path.join(run_dir, "deploy_codes.npz")
    meta = json.load(open(os.path.join(run_dir, "deploy_meta.json"), encoding="utf-8"))
    bits = meta["p"] + 1 if meta["skip_zero"] else int(np.ceil(np.log2(2 ** (meta["p"] + 1) + 1)))
    blob = np.load(npz)
    total, ok, raw_bytes, packed_bytes = 0, 0, 0, 0
    for key in sorted((k for k in blob.files if k.startswith("codes_")), key=lambda s: int(s.split("_")[1])):
        codes = blob[key].astype(np.int64)
        packed = pack_weights(codes, bits)
        back = unpack_weights(packed, codes.size, bits)
        total += 1
        ok += int(np.array_equal(back, codes))
        raw_bytes += codes.size
        packed_bytes += len(packed)
        if not np.array_equal(back, codes):
            print(f"  ! слой {key}: расхождение при распаковке")
    return {"bits": bits, "layers": total, "layers_exact": ok,
            "raw_bytes": raw_bytes, "packed_bytes": packed_bytes}


def size_table(model, cfg: dict, run_dir: str, packed: dict) -> list[tuple[str, int]]:
    theory = eval_lib.export_accounting(model)
    uncommpressed = eval_lib.worst_case_bytes(model)
    rows = [
        ("коды без упаковки (1 байт/вес) + fp32 прочее", uncommpressed["bytes_file"]),
        ("теория: коды p+1 бит + fp16-масштабы + fp32 прочее", int(theory["bytes_theory"])),
        ("реальный .bin (масштабы fp16)", packed["bytes_file"]),
        ("реальный .bin (масштабы fp32)", packed["bytes_file_fp32_scales"]),
        ("полный fp32", int(theory["bytes_fp32_same_model"])),
    ]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="каталог запуска, напр. runs/C_p4_skipzero")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ppl", action="store_true", help="не пересчитывать perplexity")
    args = ap.parse_args()

    run_dir = args.run.rstrip("/\\")
    cfg = train_lib.load_config(args.config)
    device = torch.device(args.device)

    print(f"== {run_dir} ==")
    if not os.path.exists(os.path.join(run_dir, "deploy_codes.npz")):
        print("нет deploy_codes.npz — сначала запустите src/evaluate.py --run", run_dir)
        return

    # 1) коды
    stat = verify_codes(run_dir)
    print(f"бит на код: {stat['bits']}")
    print(f"слоёв: {stat['layers']}, распаковано без потерь: {stat['layers_exact']}/{stat['layers']}")
    print(f"коды: {stat['raw_bytes']:,} байт (uint8) -> {stat['packed_bytes']:,} байт (плотный поток), "
          f"экономия {stat['raw_bytes'] / max(stat['packed_bytes'], 1):.2f}×")

    # 2) файл модели
    model, ck, _ = eval_lib.load_run(run_dir, device)
    path = os.path.join(run_dir, "weights_packed.bin")
    packed = pack_model(model, path, scale_dtype="float16")
    packed["bytes_file_fp32_scales"] = packed["bytes_file"] + packed["scales_bytes"]

    # 3) разбор обратно: коды и веса
    header, state = unpack_model(path)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path32 = os.path.join(tmp, "m32.bin")
        pack_model(model, path32, scale_dtype="float32")
        _, state32 = unpack_model(path32)
    codes_ok, max_err16, exact32 = 0, 0.0, 0
    for name, param in model.named_parameters():
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        mod = dict(model.named_modules()).get(base) if base else None
        if base and getattr(mod, "is_quant", False):
            q, k, _, _ = mod.quantizer.quantize_details(param.detach())
            ref = q.detach().cpu()
            got = state[name]
            codes_ok += int(torch.equal(state32[name], ref))
            max_err16 = max(max_err16, (got - ref).abs().max().item())
    n_quant = sum(1 for m in model.modules() if getattr(m, "is_quant", False))
    print(f"веса: восстановлено побитово {codes_ok}/{n_quant} слоёв при fp32-масштабах"
          f" (fp16-масштаб даёт расхождение {max_err16:g} на весах)")

    # 4) качество восстановленной модели
    if not args.no_ppl:
        mcfg = ModelConfig(**{**vars(model.cfg)})
        recon = TinyGPT(mcfg).to(device)
        recon.load_state_dict(state)
        recon.eval()
        vloss, vppl = eval_lib.val_ppl(recon, run_dir, cfg, device)
        _, ck2, _ = eval_lib.load_run(run_dir, device)
        print(f"perplexity: исходная {ck['val_ppl']:.3f}, из .bin {vppl:.3f} "
              f"(разница {abs(vppl - ck['val_ppl']) / ck['val_ppl'] * 100:.3f}%)")

    # 5) размеры
    print("\nразмер:")
    rows = size_table(model, cfg, run_dir, packed)
    base = rows[-1][1]
    for label, size in rows:
        print(f"  {label:58s} {size:>10,} байт  ({base / size:5.2f}×)")
    print(f"\nфайл: {path}")


if __name__ == "__main__":
    main()
