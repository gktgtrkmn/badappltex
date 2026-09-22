from __future__ import annotations

import random
import unittest

import numpy as np

from badappltex.masking import Silhouette
from badappltex.placement import (
    FormulaPlacer,
    PlacementConfig,
    compose_on_white,
)
from badappltex.rendering import RenderedFormula


def _frozen(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _silhouette(mask: list[list[bool]]) -> Silhouette:
    pixels = np.array(mask, dtype=np.bool_)
    return Silhouette(mask=_frozen(pixels))  # type: ignore[arg-type]


def _rendered(
    alpha: list[list[int]],
    *,
    latex: str = "x",
    core_threshold: int = 128,
) -> RenderedFormula:
    alpha_array = np.array(alpha, dtype=np.uint8)
    support = alpha_array > 0
    core = alpha_array >= core_threshold
    return RenderedFormula(
        latex=latex,
        alpha=_frozen(alpha_array),  # type: ignore[arg-type]
        support_mask=_frozen(support),  # type: ignore[arg-type]
        core_mask=_frozen(core),  # type: ignore[arg-type]
        baseline_depth=0,
    )


class PlacementConfigTests(unittest.TestCase):
    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PlacementConfig(attempts_per_formula=0)
        with self.assertRaises(ValueError):
            PlacementConfig(gap_px=-1)
        with self.assertRaises(TypeError):
            PlacementConfig(largest_first=1)  # type: ignore[arg-type]


class FormulaPlacerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.placer = FormulaPlacer()

    def test_all_ink_is_strictly_inside_silhouette(self) -> None:
        silhouette = _silhouette(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ]
        )
        formula = _rendered([[255, 255], [255, 255]])

        result = self.placer.place(
            silhouette,
            [formula],
            random.Random(3),
            PlacementConfig(gap_px=0),
        )

        self.assertEqual(len(result.placements), 1)
        self.assertEqual(result.outside_ink_pixels, 0)
        self.assertTrue(np.all(silhouette.mask[result.support_mask]))

    def test_formula_that_cannot_fit_is_rejected(self) -> None:
        silhouette = _silhouette([[True, True], [True, True]])
        too_wide = _rendered([[255, 255, 255]])

        result = self.placer.place(
            silhouette,
            [too_wide],
            random.Random(0),
        )

        self.assertEqual(result.placements, ())
        self.assertEqual(result.rejected_formulas, 1)

    def test_formulas_do_not_overlap(self) -> None:
        silhouette = _silhouette([[True]])
        formulas = [_rendered([[255]], latex=str(index)) for index in range(2)]

        result = self.placer.place(
            silhouette,
            formulas,
            random.Random(0),
            PlacementConfig(gap_px=0),
        )

        self.assertEqual(len(result.placements), 1)
        self.assertEqual(result.rejected_formulas, 1)

    def test_gap_separates_formula_support_pixels(self) -> None:
        silhouette = _silhouette([[True] * 5 for _ in range(5)])
        formulas = [_rendered([[255]], latex=str(index)) for index in range(8)]

        result = self.placer.place(
            silhouette,
            formulas,
            random.Random(11),
            PlacementConfig(gap_px=1, attempts_per_formula=25),
        )

        positions = [(placed.x, placed.y) for placed in result.placements]
        self.assertGreater(len(positions), 1)
        for index, (x1, y1) in enumerate(positions):
            for x2, y2 in positions[index + 1 :]:
                self.assertGreater(max(abs(x1 - x2), abs(y1 - y2)), 1)

    def test_largest_first_preserves_room_for_large_formula(self) -> None:
        silhouette = _silhouette([[True, True]])
        small = _rendered([[255]], latex="small")
        large = _rendered([[255, 255]], latex="large")

        result = self.placer.place(
            silhouette,
            [small, large],
            random.Random(0),
            PlacementConfig(gap_px=0, largest_first=True),
        )

        self.assertEqual(len(result.placements), 1)
        self.assertEqual(result.placements[0].rendered.latex, "large")

    def test_seeded_placement_is_reproducible(self) -> None:
        silhouette = _silhouette([[True] * 8 for _ in range(6)])
        formulas = [_rendered([[255]], latex=str(index)) for index in range(12)]
        config = PlacementConfig(gap_px=1)

        first = self.placer.place(silhouette, formulas, random.Random(42), config)
        second = self.placer.place(silhouette, formulas, random.Random(42), config)

        first_positions = [(item.x, item.y) for item in first.placements]
        second_positions = [(item.x, item.y) for item in second.placements]
        self.assertEqual(first_positions, second_positions)
        np.testing.assert_array_equal(first.alpha, second.alpha)

    def test_metrics_and_white_composition_are_exact(self) -> None:
        silhouette = _silhouette([[True, True]])
        formula = _rendered([[255, 128]])

        result = self.placer.place(
            silhouette,
            [formula],
            random.Random(0),
            PlacementConfig(gap_px=0),
        )
        image = compose_on_white(result)

        self.assertEqual(result.support_coverage_fraction, 1.0)
        self.assertEqual(result.core_coverage_fraction, 1.0)
        self.assertEqual(result.outside_ink_pixels, 0)
        np.testing.assert_array_equal(image, np.array([[0, 127]], dtype=np.uint8))

    def test_result_arrays_and_composition_are_immutable(self) -> None:
        silhouette = _silhouette([[True]])
        result = self.placer.place(
            silhouette,
            [_rendered([[255]])],
            random.Random(0),
        )

        for array in (
            result.alpha,
            result.support_mask,
            result.core_mask,
            compose_on_white(result),
        ):
            with self.subTest(dtype=array.dtype):
                with self.assertRaises(ValueError):
                    array.setflags(write=True)

    def test_empty_target_has_zero_coverage(self) -> None:
        silhouette = _silhouette([[False, False]])
        result = self.placer.place(
            silhouette,
            [_rendered([[255]])],
            random.Random(0),
        )

        self.assertEqual(result.support_coverage_fraction, 0.0)
        self.assertEqual(result.core_coverage_fraction, 0.0)
        self.assertEqual(result.rejected_formulas, 1)

    def test_alpha_outside_declared_support_is_rejected(self) -> None:
        invalid = RenderedFormula(
            latex="invalid",
            alpha=np.array([[255]], dtype=np.uint8),
            support_mask=np.array([[False]], dtype=np.bool_),
            core_mask=np.array([[False]], dtype=np.bool_),
            baseline_depth=0,
        )

        with self.assertRaises(ValueError):
            self.placer.place(
                _silhouette([[True]]),
                [invalid],
                random.Random(0),
            )


if __name__ == "__main__":
    unittest.main()
