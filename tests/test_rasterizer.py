from __future__ import annotations

import random
import unittest

import numpy as np
import sympy as sp

from badappltex.formulas import ArithmeticFamily, Formula
from badappltex.rendering import (
    FormulaRenderError,
    MathTextRasterizer,
    RenderStyle,
)


def _formula(latex: str) -> Formula:
    return Formula(
        lhs=sp.Integer(1),
        rhs=sp.Integer(1),
        latex=latex,
        family="test",
        complexity=0,
    )


class RenderStyleTests(unittest.TestCase):
    def test_invalid_values_are_rejected(self) -> None:
        for invalid_size in (0, -1, float("inf"), float("nan")):
            with self.subTest(font_size_pt=invalid_size):
                with self.assertRaises(ValueError):
                    RenderStyle(font_size_pt=invalid_size)

        with self.assertRaises(ValueError):
            RenderStyle(font_size_pt=12, dpi=0)
        with self.assertRaises(ValueError):
            RenderStyle(font_size_pt=12, padding_px=-1)
        with self.assertRaises(ValueError):
            RenderStyle(font_size_pt=12, core_threshold=0)
        with self.assertRaises(ValueError):
            RenderStyle(font_size_pt=12, core_threshold=256)


class MathTextRasterizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rasterizer = MathTextRasterizer()
        self.style = RenderStyle(font_size_pt=16, padding_px=2)

    def test_render_returns_nonempty_consistent_masks(self) -> None:
        rendered = self.rasterizer.render(
            _formula(r"\frac{2}{6} = \frac{1}{3}"),
            self.style,
        )

        self.assertEqual(rendered.alpha.dtype, np.uint8)
        self.assertEqual(rendered.support_mask.dtype, np.bool_)
        self.assertEqual(rendered.core_mask.dtype, np.bool_)
        self.assertEqual(rendered.alpha.shape, rendered.support_mask.shape)
        self.assertEqual(rendered.alpha.shape, rendered.core_mask.shape)
        self.assertEqual(rendered.width, rendered.alpha.shape[1])
        self.assertEqual(rendered.height, rendered.alpha.shape[0])
        self.assertGreater(np.count_nonzero(rendered.support_mask), 0)
        self.assertGreater(np.count_nonzero(rendered.core_mask), 0)
        self.assertTrue(np.all(~rendered.core_mask | rendered.support_mask))

    def test_padding_is_transparent_and_outside_support(self) -> None:
        rendered = self.rasterizer.render(_formula("2 + 2 = 4"), self.style)
        padding = self.style.padding_px

        self.assertFalse(np.any(rendered.alpha[:padding, :]))
        self.assertFalse(np.any(rendered.alpha[-padding:, :]))
        self.assertFalse(np.any(rendered.alpha[:, :padding]))
        self.assertFalse(np.any(rendered.alpha[:, -padding:]))
        self.assertFalse(np.any(rendered.support_mask[:padding, :]))

    def test_arrays_are_immutable(self) -> None:
        rendered = self.rasterizer.render(_formula("3 + 4 = 7"), self.style)

        for array in (
            rendered.alpha,
            rendered.support_mask,
            rendered.core_mask,
        ):
            with self.subTest(dtype=array.dtype):
                with self.assertRaises(ValueError):
                    array[0, 0] = 1
                with self.assertRaises(ValueError):
                    array.setflags(write=True)

    def test_identical_request_returns_cached_object(self) -> None:
        formula = _formula("5 - 2 = 3")

        first = self.rasterizer.render(formula, self.style)
        second = self.rasterizer.render(formula, self.style)

        self.assertIs(first, second)
        self.assertEqual(self.rasterizer.cache_size, 1)

    def test_cache_is_bounded_and_uses_lru_eviction(self) -> None:
        rasterizer = MathTextRasterizer(max_cache_entries=2)
        first_formula = _formula("1 + 1 = 2")
        second_formula = _formula("2 + 2 = 4")
        first = rasterizer.render(first_formula, self.style)
        second = rasterizer.render(second_formula, self.style)
        rasterizer.render(first_formula, self.style)  # Make first most recent.
        rasterizer.render(_formula("3 + 3 = 6"), self.style)

        self.assertEqual(rasterizer.cache_size, 2)
        self.assertIs(first, rasterizer.render(first_formula, self.style))

        evicted = rasterizer.render(second_formula, self.style)
        self.assertEqual(rasterizer.cache_size, 2)
        self.assertIsNot(second, evicted)

    def test_different_threshold_reuses_pixels_but_not_cache_entry(self) -> None:
        formula = _formula("6 + 1 = 7")
        low = self.rasterizer.render(
            formula,
            RenderStyle(font_size_pt=16, core_threshold=32),
        )
        high = self.rasterizer.render(
            formula,
            RenderStyle(font_size_pt=16, core_threshold=224),
        )

        np.testing.assert_array_equal(low.alpha, high.alpha)
        self.assertGreaterEqual(
            np.count_nonzero(low.core_mask),
            np.count_nonzero(high.core_mask),
        )
        self.assertIsNot(low, high)

    def test_malformed_and_empty_latex_raise_render_error(self) -> None:
        for latex in ("", r"\notacommand{2}", "$2 + 2 = 4$"):
            with self.subTest(latex=latex):
                with self.assertRaises(FormulaRenderError):
                    self.rasterizer.render(_formula(latex), self.style)

    def test_generated_arithmetic_formulas_are_renderable(self) -> None:
        family = ArithmeticFamily()
        rng = random.Random(20260921)

        for _ in range(100):
            formula = family.generate(rng)
            rendered = self.rasterizer.render(formula, self.style)
            self.assertGreater(rendered.width, 0, formula.latex)
            self.assertGreater(rendered.height, 0, formula.latex)


if __name__ == "__main__":
    unittest.main()
