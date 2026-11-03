"""Упаковка кодов k в плотный битовый поток и сборка бинарных файлов модели.

Два формата:

* **SZP1** (плотный) — все коды `p+1` бит подряд, без выравнивания на байт:
  8 кодов при p = 4 (5 бит) занимают ровно 5 байт.
* **SZM1** (с маской нулей, «Step Zero Mask») — для сетки с нулём (вариант B):
  1 бит маски на вес (`0` — ровно ноль, `1` — ненулевой), затем коды только
  ненулевых весов (p+1 бит на код), затем масштабы блоков.

Общее для обоих форматов:
    "MAGIC" | uint32 LE (длина JSON) | JSON-заголовок | payload
Масштабы — один на блок (fp16 по теории, разд. 11; fp32 — для побитовой точности).
Эмбеддинги, LayerNorm, bias и lm_head хранятся в fp32.
"""
from __future__ import annotations

import json
import math
import struct

import numpy as np
import torch

from quant import QuantLinear, encode_k, encode_k_with_zero

MAGIC = b"SZP1"
MAGIC_SPARSE = b"SZM1"
MAX_BITS = 24


# ---------------------------------------------------------------------------
# Битовая упаковка кодов
# ---------------------------------------------------------------------------
def _check_bits(bits_per_code: int) -> int:
    bits = int(bits_per_code)
    if not (1 <= bits <= MAX_BITS):
        raise ValueError(f"bits_per_code должен быть в 1..{MAX_BITS}, получено {bits_per_code!r}")
    return bits


def _as_codes(codes) -> np.ndarray:
    if isinstance(codes, torch.Tensor):
        codes = codes.detach().cpu().numpy()
    arr = np.asarray(codes)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    return arr.reshape(-1).astype(np.int64, copy=False)


def pack_weights(codes, bits_per_code: int) -> bytes:
    """Упаковать коды (неотрицательные целые < 2^bits) в плотный поток байт.

    Биты пишутся от младшего к старшему внутри кода и подряд в байтах
    (bitorder='little'), хвост последнего байта добивается нулями.
    """
    bits = _check_bits(bits_per_code)
    arr = _as_codes(codes)
    if arr.size and (arr.min() < 0 or arr.max() > (1 << bits) - 1):
        raise ValueError(f"код вне диапазона 0..{(1 << bits) - 1} для {bits} бит")
    if arr.size == 0:
        return b""
    shifts = np.arange(bits, dtype=np.int64)
    bitmat = ((arr[:, None] >> shifts[None, :]) & 1).astype(np.uint8)  # (n, bits)
    flat = bitmat.reshape(-1)
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(flat, bitorder="little").tobytes()


def unpack_weights(byte_stream: bytes, num_elements: int, bits_per_code: int) -> np.ndarray:
    """Обратное к pack_weights: восстановить ровно num_elements кодов."""
    bits = _check_bits(bits_per_code)
    n = int(num_elements)
    need_bits = n * bits
    data = np.frombuffer(bytes(byte_stream), dtype=np.uint8)
    if data.size * 8 < need_bits:
        raise ValueError(f"поток короткий: {data.size * 8} бит < {need_bits}")
    flat = np.unpackbits(data, bitorder="little")[:need_bits]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    bitmat = flat.reshape(n, bits).astype(np.int64)
    return (bitmat << np.arange(bits, dtype=np.int64)[None, :]).sum(axis=1)


def pack_mask(mask) -> bytes:
    """Битовая маска нулей: 8 весов на байт (bitorder='little')."""
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    return np.packbits(arr.astype(np.uint8), bitorder="little").tobytes()


def unpack_mask(byte_stream: bytes, num_elements: int) -> np.ndarray:
    data = np.frombuffer(bytes(byte_stream), dtype=np.uint8)
    bits = np.unpackbits(data, bitorder="little")[: int(num_elements)]
    return bits.astype(bool)


# ---------------------------------------------------------------------------
# Масштабы: один на блок, а не на вес
# ---------------------------------------------------------------------------
def block_scales(weight: torch.Tensor, quantizer) -> np.ndarray:
    if quantizer.scale_mode == "global":
        return np.zeros(0, dtype=np.float32)
    with torch.no_grad():
        scale = quantizer.compute_scale(weight.detach())
    flat = scale.reshape(-1).cpu().numpy()
    return flat[:: quantizer.block_size].astype(np.float32)


def expand_block_scales(blocks: np.ndarray, shape, block_size: int) -> torch.Tensor:
    numel = int(np.prod(shape))
    if blocks.size == 0:
        return torch.ones(shape, dtype=torch.float32)
    t = torch.from_numpy(np.asarray(blocks).astype(np.float32))
    full = torch.repeat_interleave(t, block_size)[:numel]
    return full.reshape(shape)


def bits_for_codes(quantizer) -> int:
    """Сколько бит на код: p+1 без нуля, log2(2^(p+1)+1) с нулём."""
    if quantizer.skip_zero:
        return quantizer.p + 1
    return int(math.ceil(math.log2((1 << (quantizer.p + 1)) + 1)))


def _decode_codes_np(codes: np.ndarray, p: int, skip_zero: bool) -> np.ndarray:
    shift = 2 ** p
    c = codes.astype(np.int64)
    if not skip_zero:
        return c - shift
    return np.where(c <= shift - 1, c - shift, c - shift + 1)


# ---------------------------------------------------------------------------
# Обход параметров (с учётом tied-весов)
# ---------------------------------------------------------------------------
def _iter_state(model):
    """(name, tensor, quant_module|None, alias_ref|None) по всему state_dict."""
    quant_mods = {n: m for n, m in model.named_modules() if getattr(m, "is_quant", False)}
    seen: dict[int, str] = {}
    for name, tensor in model.state_dict().items():
        ptr = tensor.data_ptr()
        if ptr in seen:
            yield name, tensor, None, seen[ptr]
            continue
        seen[ptr] = name
        mod = quant_mods.get(name[: -len(".weight")]) if name.endswith(".weight") else None
        yield name, tensor, mod, None


def _np_dtype(scale_dtype: str):
    if scale_dtype not in ("float16", "float32"):
        raise ValueError("scale_dtype: float16 | float32")
    return np.float16 if scale_dtype == "float16" else np.float32


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic not in (MAGIC, MAGIC_SPARSE):
            raise ValueError(f"{path}: неизвестный формат {magic!r}")
        (hlen,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(hlen).decode("utf-8"))
    header["_magic"] = magic.decode()
    return header


def _payload_stream(f):
    f.seek(4)
    (hlen,) = struct.unpack("<I", f.read(4))
    header = json.loads(f.read(hlen).decode("utf-8"))
    return header


# ---------------------------------------------------------------------------
# Формат SZP1: плотный
# ---------------------------------------------------------------------------
@torch.no_grad()
def pack_model(model, path: str, scale_dtype: str = "float16", include_fp32: bool = True) -> dict:
    """Записать модель в плотный .bin (коды p+1 бит, масштабы fp16/fp32, прочее fp32)."""
    np_scale = _np_dtype(scale_dtype)
    q = getattr(model, "quantizer", None)

    layers, payload = [], bytearray()
    codes_total = scales_total = fp32_total = 0
    for name, param, mod, ref in _iter_state(model):
        if ref is not None:
            layers.append({"name": name, "kind": "alias", "ref": ref, "shape": list(param.shape)})
            continue
        if mod is not None and q is not None:
            _, k, _, _ = mod.quantizer.quantize_details(param.detach())
            codes = encode_k(k, q.p) if q.skip_zero else encode_k_with_zero(k, q.p)
            bits = bits_for_codes(q)
            codes_bytes = pack_weights(codes, bits)
            scales = block_scales(param, mod.quantizer)
            scale_bytes = scales.astype(np_scale).tobytes() if scales.size else b""
            payload += codes_bytes + scale_bytes
            codes_total += len(codes_bytes)
            scales_total += len(scale_bytes)
            layers.append({"name": name, "kind": "quant", "shape": list(param.shape),
                           "numel": int(param.numel()), "bits": bits,
                           "codes_bytes": len(codes_bytes), "scale_bytes": len(scale_bytes),
                           "scales": int(scales.size), "block_size": q.block_size,
                           "scale_mode": q.scale_mode, "scale_dtype": scale_dtype})
        else:
            if not include_fp32:
                continue
            raw = param.detach().float().contiguous().cpu().numpy().astype(np.float32)
            blob = raw.tobytes()
            payload += blob
            fp32_total += len(blob)
            layers.append({"name": name, "kind": "fp32", "shape": list(param.shape),
                           "numel": int(param.numel()), "bytes": len(blob)})

    header = {
        "magic": MAGIC.decode(),
        "quant": None if q is None else {"p": q.p, "n": q.n, "skip_zero": q.skip_zero,
                                        "scale_mode": q.scale_mode, "block_size": q.block_size,
                                        "step": q.step},
        "model_cfg": {k: v for k, v in vars(model.cfg).items()},
        "layers": layers,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(bytes(payload))

    return {
        "path": path,
        "bytes_file": len(MAGIC) + 4 + len(header_bytes) + len(payload),
        "header_bytes": len(header_bytes),
        "codes_bytes": codes_total,
        "scales_bytes": scales_total,
        "fp32_bytes": fp32_total,
        "scale_dtype": scale_dtype,
        "bits_per_code": None if q is None else bits_for_codes(q),
        "quant_weights": None if q is None else model.n_quant_params(),
        "params_total": int(sum(p.numel() for p in model.parameters())),
        "layers": len(layers),
    }


# ---------------------------------------------------------------------------
# Формат SZM1: маска нулей + коды только ненулевых
# ---------------------------------------------------------------------------
@torch.no_grad()
def pack_sparse(model, path: str, scale_dtype: str = "float16") -> dict:
    """Записать модель с битовой маской нулей (нули не занимают кодов).

    Маска: 1 бит на вес, 1 = ненулевой. Коды (p+1 бит) пишутся только для
    ненулевых весов, поэтому ноль в сетке становится бесплатным по кодам.
    """
    np_scale = _np_dtype(scale_dtype)
    q = getattr(model, "quantizer", None)
    if q is None:
        raise ValueError("sparse-формат имеет смысл только для квантованной модели")

    layers, payload = [], bytearray()
    mask_total = codes_total = scales_total = fp32_total = 0
    nonzero_total = zero_total = 0
    for name, param, mod, ref in _iter_state(model):
        if ref is not None:
            layers.append({"name": name, "kind": "alias", "ref": ref, "shape": list(param.shape)})
            continue
        if mod is not None:
            _, k, _, _ = mod.quantizer.quantize_details(param.detach())
            k_np = k.detach().reshape(-1).cpu().numpy()
            mask = k_np != 0
            mask_bytes = pack_mask(mask)
            codes = encode_k(torch.from_numpy(k_np[mask]), q.p)  # ненулевые -> 0..2^(p+1)-1
            bits = q.p + 1
            codes_bytes = pack_weights(codes, bits)
            scales = block_scales(param, mod.quantizer)
            scale_bytes = scales.astype(np_scale).tobytes() if scales.size else b""
            payload += mask_bytes + codes_bytes + scale_bytes
            mask_total += len(mask_bytes)
            codes_total += len(codes_bytes)
            scales_total += len(scale_bytes)
            nz = int(mask.sum())
            nonzero_total += nz
            zero_total += int(mask.size - nz)
            layers.append({"name": name, "kind": "quant_sparse", "shape": list(param.shape),
                           "numel": int(param.numel()), "nonzero": nz,
                           "bits": bits, "mask_bytes": len(mask_bytes),
                           "codes_bytes": len(codes_bytes), "scale_bytes": len(scale_bytes),
                           "scales": int(scales.size), "block_size": q.block_size,
                           "scale_mode": q.scale_mode, "scale_dtype": scale_dtype})
        else:
            raw = param.detach().float().contiguous().cpu().numpy().astype(np.float32)
            blob = raw.tobytes()
            payload += blob
            fp32_total += len(blob)
            layers.append({"name": name, "kind": "fp32", "shape": list(param.shape),
                           "numel": int(param.numel()), "bytes": len(blob)})

    header = {
        "magic": MAGIC_SPARSE.decode(),
        "quant": {"p": q.p, "n": q.n, "skip_zero": q.skip_zero,
                  "scale_mode": q.scale_mode, "block_size": q.block_size,
                  "step": q.step, "bits": q.p + 1},
        "model_cfg": {k: v for k, v in vars(model.cfg).items()},
        "sparsity": {"nonzero": nonzero_total, "zero": zero_total,
                     "zero_frac": zero_total / max(nonzero_total + zero_total, 1)},
        "layers": layers,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC_SPARSE)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(bytes(payload))

    numel = nonzero_total + zero_total
    return {
        "path": path,
        "bytes_file": len(MAGIC_SPARSE) + 4 + len(header_bytes) + len(payload),
        "header_bytes": len(header_bytes),
        "mask_bytes": mask_total,
        "codes_bytes": codes_total,
        "scales_bytes": scales_total,
        "fp32_bytes": fp32_total,
        "nonzero": nonzero_total,
        "zero": zero_total,
        "zero_frac": zero_total / max(numel, 1),
        "bits_per_code": q.p + 1,
        "scale_dtype": scale_dtype,
        "params_total": int(sum(p.numel() for p in model.parameters())),
        "layers": len(layers),
    }


# ---------------------------------------------------------------------------
# Чтение
# ---------------------------------------------------------------------------
def unpack_model(path: str) -> tuple[dict, dict[str, torch.Tensor]]:
    """Прочитать плотный .bin и восстановить словарь параметров."""
    header = read_header(path)
    if header["_magic"] != MAGIC.decode():
        raise ValueError(f"{path}: ожидался плотный формат SZP1")
    q = header["quant"]
    state: dict[str, torch.Tensor] = {}
    aliases: list[tuple[str, str]] = []
    with open(path, "rb") as f:
        header = _payload_stream(f)
        for layer in header["layers"]:
            shape = tuple(layer["shape"])
            if layer["kind"] == "alias":
                aliases.append((layer["name"], layer["ref"]))
                continue
            numel = int(layer["numel"])
            if layer["kind"] == "quant":
                bits = int(layer["bits"])
                nb = (numel * bits + 7) // 8
                codes = unpack_weights(f.read(nb), numel, bits)
                if q["skip_zero"]:
                    k = _decode_codes_np(codes, q["p"], skip_zero=True)
                    assert (k != 0).all(), "в модели C найден нулевой код"
                else:
                    k = _decode_codes_np(codes, q["p"], skip_zero=False)
                np_dtype = np.float16 if layer["scale_dtype"] == "float16" else np.float32
                sc_bytes = int(layer.get("scale_bytes", 0))
                scales = (np.frombuffer(f.read(sc_bytes), dtype=np_dtype).astype(np.float32)
                          if sc_bytes else np.zeros(0, dtype=np.float32))
                scale = expand_block_scales(scales, shape, int(layer.get("block_size", 32)))
                if q["scale_mode"] == "global":
                    scale = torch.ones(shape, dtype=torch.float32)
                w = torch.from_numpy(k.reshape(shape).astype(np.float32)) * float(q["step"]) * scale
                state[layer["name"]] = w
            else:
                raw = np.frombuffer(f.read(int(layer["bytes"])), dtype=np.float32)
                state[layer["name"]] = torch.from_numpy(raw.reshape(shape).copy())
    for name, ref in aliases:
        state[name] = state[ref]
    return header, state


def unpack_sparse(path: str) -> tuple[dict, dict[str, torch.Tensor]]:
    """Прочитать .bin с маской нулей и восстановить параметры (нули — ровно 0.0)."""
    header = read_header(path)
    if header["_magic"] != MAGIC_SPARSE.decode():
        raise ValueError(f"{path}: ожидался разреженный формат SZM1")
    q = header["quant"]
    state: dict[str, torch.Tensor] = {}
    aliases: list[tuple[str, str]] = []
    with open(path, "rb") as f:
        header = _payload_stream(f)
        for layer in header["layers"]:
            shape = tuple(layer["shape"])
            if layer["kind"] == "alias":
                aliases.append((layer["name"], layer["ref"]))
                continue
            numel = int(layer["numel"])
            if layer["kind"] == "quant_sparse":
                nz = int(layer["nonzero"])
                mask = unpack_mask(f.read(int(layer["mask_bytes"])), numel)
                if int(mask.sum()) != nz:
                    raise ValueError(f"{layer['name']}: маска не совпала с числом ненулевых")
                codes = unpack_weights(f.read(int(layer["codes_bytes"])), nz, int(layer["bits"]))
                k = np.zeros(numel, dtype=np.int64)
                if nz:
                    k[mask] = _decode_codes_np(codes, q["p"], skip_zero=not q["skip_zero"])
                assert nz == 0 or (k[mask] != 0).all(), "ноль попал в ненулевые"
                np_dtype = np.float16 if layer["scale_dtype"] == "float16" else np.float32
                sc_bytes = int(layer.get("scale_bytes", 0))
                scales = (np.frombuffer(f.read(sc_bytes), dtype=np_dtype).astype(np.float32)
                          if sc_bytes else np.zeros(0, dtype=np.float32))
                scale = expand_block_scales(scales, shape, int(layer.get("block_size", 32)))
                if q["scale_mode"] == "global":
                    scale = torch.ones(shape, dtype=torch.float32)
                w = torch.from_numpy(k.reshape(shape).astype(np.float32)) * float(q["step"]) * scale
                state[layer["name"]] = w
            else:
                raw = np.frombuffer(f.read(int(layer["bytes"])), dtype=np.float32)
                state[layer["name"]] = torch.from_numpy(raw.reshape(shape).copy())
    for name, ref in aliases:
        state[name] = state[ref]
    return header, state
