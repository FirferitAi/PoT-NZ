"""Проверки упаковщика: точный размер, побитовый roundtrip, сборка .bin модели."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ModelConfig, TinyGPT  # noqa: E402
from pack import read_header  # noqa: E402
from quant import QuantLinear  # noqa: E402
from pack import (  # noqa: E402
    bits_for_codes,
    expand_block_scales,
    pack_mask,
    pack_model,
    pack_sparse,
    pack_weights,
    unpack_mask,
    unpack_model,
    unpack_sparse,
    unpack_weights,
)


class TestBitPacking(unittest.TestCase):
    def test_example_5_bits_8_codes(self):
        codes = np.array([16, 0, 31, 15, 1, 30, 8, 23], dtype=np.int64)
        blob = pack_weights(codes, 5)
        self.assertEqual(len(blob), 5, "8 кодов по 5 бит = 40 бит = ровно 5 байт")
        self.assertTrue(np.array_equal(unpack_weights(blob, 8, 5), codes))

    def test_roundtrip_all_bit_widths(self):
        rng = np.random.default_rng(0)
        for bits in range(1, 10):
            n = 1000
            codes = rng.integers(0, 1 << bits, size=n, dtype=np.int64)
            blob = pack_weights(codes, bits)
            self.assertEqual(len(blob), (n * bits + 7) // 8, f"bits={bits}: размер не минимален")
            out = unpack_weights(blob, n, bits)
            self.assertTrue(np.array_equal(out, codes), f"bits={bits}: потеря при roundtrip")

    def test_empty_and_padding(self):
        self.assertEqual(pack_weights(np.zeros(0, dtype=np.int64), 5), b"")
        codes = np.array([1, 2, 3], dtype=np.int64)
        blob = pack_weights(codes, 3)  # 9 бит -> 2 байта, 7 бит паддинга
        self.assertEqual(len(blob), 2)
        self.assertTrue(np.array_equal(unpack_weights(blob, 3, 3), codes))

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            pack_weights(np.array([32]), 5)
        with self.assertRaises(ValueError):
            pack_weights(np.array([-1]), 5)
        with self.assertRaises(ValueError):
            unpack_weights(b"\x00", 100, 5)


class TestBlockScales(unittest.TestCase):
    def test_expand_matches_weight_shape(self):
        from quant import StepQuantizer

        q = StepQuantizer(n=1.0, p=4, skip_zero=True, scale_mode="per_block", block_size=4)
        w = torch.arange(1, 11, dtype=torch.float32).view(2, 5)  # numel 10, 3 блока (4,4,2)
        blocks = np.array([4.0, 8.0, 10.0], dtype=np.float32)  # absmax блоков (4,4,2)
        scale = expand_block_scales(blocks, w.shape, 4)
        self.assertEqual(tuple(scale.shape), tuple(w.shape))
        self.assertTrue(torch.allclose(scale.reshape(-1), q.compute_scale(w).reshape(-1)))


class TestBinFile(unittest.TestCase):
    def _model(self, p=4):
        cfg = ModelConfig(vocab_size=17, d_model=16, n_heads=2, n_layers=1, d_ff=32,
                          context=8, dropout=0.0,
                          quant={"n": 1.0, "p": p, "skip_zero": True,
                                 "scale_mode": "per_block", "block_size": 8})
        torch.manual_seed(0)
        return TinyGPT(cfg).eval()

    def test_bin_roundtrip(self):
        model = self._model()
        self.assertEqual(bits_for_codes(model.quantizer), 5)
        qmods = {n: m for n, m in model.named_modules() if hasattr(m, "quantizer")}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.bin")
            info = pack_model(model, path, scale_dtype="float32")
            self.assertEqual(os.path.getsize(path), info["bytes_file"])
            header, state = unpack_model(path)
            self.assertEqual(header["quant"]["p"], 4)

            for name, param in model.named_parameters():
                got = state[name]
                self.assertEqual(tuple(got.shape), tuple(param.shape), name)
                base = name[: -len(".weight")] if name.endswith(".weight") else None
                if base in qmods:
                    # fp32-масштаб -> восстановленный вес побитово равен q
                    ref = qmods[base].quantizer.quantize_details(param.detach())[0]
                    self.assertTrue(torch.equal(got, ref), f"{name}: не совпало")
                else:
                    self.assertTrue(torch.equal(got, param.detach().float()), name)

    def test_with_zero_grid_needs_extra_bit(self):
        cfg = ModelConfig(vocab_size=13, d_model=16, n_heads=2, n_layers=1, d_ff=32,
                          context=8, dropout=0.0,
                          quant={"n": 1.0, "p": 4, "skip_zero": False,
                                 "scale_mode": "per_block", "block_size": 8})
        torch.manual_seed(3)
        model = TinyGPT(cfg).eval()
        # 2^(p+1)+1 = 33 значения -> нужно 6 бит, а не 5
        self.assertEqual(bits_for_codes(model.quantizer), 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.bin")
            pack_model(model, path, scale_dtype="float32")
            header, state = unpack_model(path)
            self.assertFalse(header["quant"]["skip_zero"])
            for name, mod in model.named_modules():
                if isinstance(mod, QuantLinear) and f"{name}.weight" in state:
                    ref = mod.quantizer.quantize_details(mod.weight.detach())[0]
                    self.assertTrue(torch.equal(state[f"{name}.weight"], ref), name)

    def test_fp16_scales_close(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m16.bin")
            info16 = pack_model(model, path, scale_dtype="float16")
            path32 = os.path.join(tmp, "m32.bin")
            info32 = pack_model(model, path32, scale_dtype="float32")
            self.assertLess(info16["bytes_file"], info32["bytes_file"])
            self.assertEqual(info16["codes_bytes"], info32["codes_bytes"])
            _, st16 = unpack_model(path)
            _, st32 = unpack_model(path32)
            for name in st16:
                if name.endswith(".weight") and "blocks." in name:
                    err = (st16[name] - st32[name]).abs().max().item()
                    self.assertLess(err, 1e-3, f"{name}: fp16-масштаб должен давать малую ошибку")

    def test_fp32_only_model(self):
        cfg = ModelConfig(vocab_size=11, d_model=8, n_heads=2, n_layers=1, d_ff=16,
                          context=4, dropout=0.0, quant=None)
        model = TinyGPT(cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fp32.bin")
            pack_model(model, path)
            self.assertGreater(os.path.getsize(path), 0)
            _, state = unpack_model(path)
            for name, param in model.named_parameters():
                self.assertTrue(torch.equal(state[name], param.detach().float()), name)


class TestSparse(unittest.TestCase):
    def _model(self, p=4, vocab=19):
        cfg = ModelConfig(vocab_size=vocab, d_model=16, n_heads=2, n_layers=1, d_ff=32,
                          context=8, dropout=0.0,
                          quant={"n": 1.0, "p": p, "skip_zero": False,
                                 "scale_mode": "per_block", "block_size": 8})
        torch.manual_seed(7)
        return TinyGPT(cfg).eval()

    def test_mask_roundtrip(self):
        mask = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=bool)
        blob = pack_mask(mask)
        self.assertEqual(len(blob), 2)  # 9 весов -> 2 байта
        self.assertTrue(np.array_equal(unpack_mask(blob, 9), mask))
        self.assertTrue(np.array_equal(unpack_mask(blob, 5), mask[:5]))

    def test_pack_sparse_zero_loss(self):
        model = self._model()
        qmods = {n: m for n, m in model.named_modules() if isinstance(m, QuantLinear)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.bin")
            info = pack_sparse(model, path, scale_dtype="float32")
            self.assertEqual(os.path.getsize(path), info["bytes_file"])
            header, state = unpack_sparse(path)
            self.assertEqual(header["sparsity"]["zero"],
                             int(header["sparsity"]["zero_frac"] > 0 and header["sparsity"]["zero"]))
            nz = 0
            for name, mod in qmods.items():
                w = mod.weight.detach()
                ref, k, _, _ = mod.quantizer.quantize_details(w)
                got = state[f"{name}.weight"]
                self.assertTrue(torch.equal(got, ref.detach()), name)
                self.assertTrue(torch.equal(got[k.cpu() == 0], torch.zeros_like(got[k.cpu() == 0])))
                nz += int((k != 0).sum())
            self.assertEqual(info["nonzero"], nz)

    def test_sparse_header_and_size(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as tmp:
            dense_path = os.path.join(tmp, "d.bin")
            sparse_path = os.path.join(tmp, "s.bin")
            d = pack_model(model, dense_path, scale_dtype="float32")
            sp = pack_sparse(model, sparse_path, scale_dtype="float32")
            # маска стоит 1 бит, ненулевые коды 5 бит: при большой доле нулей файл меньше
            self.assertLessEqual(sp["mask_bytes"], (sp["nonzero"] + sp["zero"] + 7) // 8)
            self.assertLessEqual(sp["codes_bytes"], d["codes_bytes"])
            header = read_header(sparse_path)
            self.assertEqual(header["_magic"], "SZM1")

    def test_tied_embeddings_alias(self):
        cfg = ModelConfig(vocab_size=23, d_model=16, n_heads=2, n_layers=1, d_ff=32,
                          context=8, dropout=0.0, tie_embeddings=True,
                          quant={"n": 1.0, "p": 4, "skip_zero": True,
                                 "scale_mode": "per_block", "block_size": 8})
        torch.manual_seed(11)
        model = TinyGPT(cfg).eval()
        self.assertIs(model.lm_head.weight, model.tok_emb.weight)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.bin")
            info = pack_model(model, path, scale_dtype="float32")
            header, state = unpack_model(path)
            self.assertEqual(set(state), set(model.state_dict()))
            self.assertIs(state["lm_head.weight"], state["tok_emb.weight"])
            self.assertLess(info["fp32_bytes"], 2 * 23 * 16 * 4)  # голова не дублируется


if __name__ == "__main__":
    unittest.main(verbosity=2)
