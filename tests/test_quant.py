"""Проверки теории: сетка без нуля, точные двоичные дроби, кодирование p+1 бит, STE."""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quant import (  # noqa: E402
    QuantLinear,
    StepQuantizer,
    bits_per_weight,
    decode_c,
    encode_k,
)


class TestGrid(unittest.TestCase):
    def test_step_is_exact_binary_fraction(self):
        for p in range(1, 9):
            q = StepQuantizer(n=1.0, p=p)
            self.assertEqual(q.step * 2 ** p, 1.0)  # точное равенство, без 1/3
            mant = q.step.as_integer_ratio()[1]
            self.assertEqual(mant & (mant - 1), 0, "знаменатель шага должен быть степенью двойки")

    def test_no_zero_and_on_grid(self):
        torch.manual_seed(0)
        q = StepQuantizer(n=1.0, p=4, skip_zero=True, scale_mode="per_block")
        w = torch.randn(64, 128)
        qq, k, u, scale = q.quantize_details(w)
        self.assertEqual(int((k == 0).sum()), 0, "в сетке без нуля не должно быть k=0")
        self.assertTrue(torch.all(k.abs() <= 2 ** 4))
        on_grid = torch.isclose(qq / scale / q.step, torch.round(qq / scale / q.step), atol=1e-5)
        self.assertTrue(bool(on_grid.all()), "все веса обязаны лежать на сетке")

    def test_levels_count(self):
        for p in (2, 3, 4, 5):
            q = StepQuantizer(n=1.0, p=p, skip_zero=True, scale_mode="global")
            w = torch.linspace(-1.0, 1.0, 4001).view(1, -1)
            qq = q(w)
            levels = torch.unique(qq).numel()
            self.assertEqual(levels, 2 ** (p + 1), f"p={p}: ожидалось 2^(p+1) значений")
            self.assertEqual(bits_per_weight(p, True), p + 1)

    def test_zero_kept_when_asked(self):
        q = StepQuantizer(n=1.0, p=4, skip_zero=False, scale_mode="global")
        w = torch.tensor([[0.0, 1e-4, -1e-4, 0.5]])
        qq = q(w)
        self.assertIn(0.0, qq.flatten().tolist())

    def test_global_clip(self):
        q = StepQuantizer(n=1.0, p=4, skip_zero=True, scale_mode="global")
        qq = q(torch.tensor([[5.0, -7.0]]))
        self.assertTrue(bool((qq.abs() <= 1.0 + 1e-9).all()))

    def test_per_block_scale_covers_block(self):
        q = StepQuantizer(n=1.0, p=4, skip_zero=True, scale_mode="per_block", block_size=8)
        w = torch.arange(1, 17, dtype=torch.float32).view(2, 8) * 0.1
        scale = q.compute_scale(w)
        self.assertEqual(scale.numel(), w.numel())
        # блоки идут по плоскому массиву: [0.1..0.8] и [0.9..1.6]
        self.assertTrue(torch.allclose(scale.view(2, 8)[:, 0], torch.tensor([0.8, 1.6])))
        qq = q(w)
        self.assertTrue(bool((qq.abs() <= w.abs().max() + 1e-6).all()))


class TestCodec(unittest.TestCase):
    def test_roundtrip_all_codes(self):
        for p in (2, 3, 4, 5):
            ks = torch.cat([torch.arange(-2 ** p, 0), torch.arange(1, 2 ** p + 1)])
            c = encode_k(ks, p)
            self.assertEqual(sorted(c.tolist()), list(range(2 ** (p + 1))), "все коды заняты")
            self.assertTrue(torch.equal(decode_c(c, p), ks))

    def test_example_from_theory(self):
        p = 4
        self.assertEqual(int(encode_k(torch.tensor(-16), p)), 0)
        self.assertEqual(int(encode_k(torch.tensor(-1), p)), 15)
        self.assertEqual(int(encode_k(torch.tensor(1), p)), 16)
        self.assertEqual(int(encode_k(torch.tensor(16), p)), 31)


class TestSTE(unittest.TestCase):
    def test_forward_quantized_and_grad_flows(self):
        torch.manual_seed(0)
        lin = QuantLinear(8, 4, StepQuantizer(n=1.0, p=4, skip_zero=True, scale_mode="per_block"))
        x = torch.randn(3, 8)
        out = lin(x)
        self.assertEqual(out.shape, (3, 4))
        out.sum().backward()
        self.assertIsNotNone(lin.weight.grad)
        self.assertGreater(float(lin.weight.grad.abs().sum()), 0.0, "STE: градиент не должен быть нулевым")

    def test_forward_uses_quantized_weight(self):
        torch.manual_seed(1)
        lin = QuantLinear(16, 16, StepQuantizer(n=1.0, p=3, skip_zero=True, scale_mode="per_block"),
                          bias=False)
        x = torch.randn(5, 16)
        got = lin(x)
        ref = x @ lin.quantized_weight().T
        self.assertTrue(torch.allclose(got, ref, atol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
