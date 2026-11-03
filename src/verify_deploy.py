"""Проверка «код -> вес» без потерь: коды + масштабы дают ровно те же веса, что в forward.

Демонстрирует свойство теории: шаг — двоичная дробь, поэтому k·step·scale
восстанавливается точно, без ошибок округления при кодировании/декодировании.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import data as data_lib  # noqa: F401  (нужен для load_run -> tokenizer)
from evaluate import load_run
from pack import expand_block_scales
from quant import decode_c, decode_c_with_zero


def main() -> None:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/C_p4_skipzero"
    device = torch.device("cpu")
    model, ck, _ = load_run(run_dir, device)
    q = model.quantizer
    npz_path = os.path.join(run_dir, "deploy_codes.npz")
    meta_path = os.path.join(run_dir, "deploy_meta.json")
    if q is None:
        print("fp32-модель: проверять нечего")
        return
    if not os.path.exists(npz_path):
        print(f"нет {npz_path}: сначала запустите src/evaluate.py --run {run_dir}")
        return

    meta = json.load(open(meta_path, encoding="utf-8"))
    blob = np.load(npz_path)
    print(f"{os.path.basename(run_dir)}: {meta['codec']}, p={meta['p']}, "
          f"масштаб={meta['scale_mode']}, блок={meta['block_size']}")

    total, exact, max_err, fp16_err, fwd_gap = 0, 0, 0.0, 0.0, 0.0
    for i, mod in enumerate(model.quant_linears()):
        codes = torch.from_numpy(blob[f"codes_{i}"].astype(np.int64))
        k = decode_c(codes, q.p) if q.skip_zero else decode_c_with_zero(codes, q.p)
        k = k.view(mod.weight.shape)
        if q.scale_mode == "global":
            scale = torch.ones(mod.weight.shape, dtype=torch.float32)
        else:
            scale = expand_block_scales(blob[f"scale_{i}"], mod.weight.shape, q.block_size)
        # эталон — сам квантованный вес q (а не STE-выход w+(q-w), который может
        # отличаться от q на 1 ulp fp32 из-за порядка операций)
        w_q, _, _, _ = mod.quantizer.quantize_details(mod.weight.detach())
        w_ref = w_q
        fwd_gap = max(fwd_gap, (mod.quantizer(mod.weight).detach() - w_q).abs().max().item())
        w_dec = (k.to(torch.float32) * mod.quantizer.step * scale)
        err = (w_dec - w_ref).abs().max().item()
        max_err = max(max_err, err)
        exact += int(torch.equal(w_dec, w_ref))
        total += 1
        if q.scale_mode != "global":
            w_fp16 = (k.to(torch.float32) * mod.quantizer.step * scale.half().float())
            fp16_err = max(fp16_err, (w_fp16 - w_ref).abs().max().item())

    print(f"слоёв: {total}, побитово совпало: {exact}/{total}, max|ошибка| = {max_err:g}")
    print(f"справка: STE-выход w+(q-w) отличается от q на {fwd_gap:g} "
          f"(порядок арифметики fp32, не кодек)")
    if q.scale_mode != "global":
        print(f"справка: если хранить масштаб в fp16 (2 байта/блок), ошибка = {fp16_err:g} "
              f"(только мантисса масштаба, коды при этом точные)")
    print("ВЫВОД:", "кодирование/декодирование точное (двоичные дроби)" if max_err == 0.0
          else "есть расхождение — проверить кодек")


if __name__ == "__main__":
    main()
