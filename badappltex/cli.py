"""Minimal command-line preview for the implemented formula pipeline."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image
from numpy.typing import NDArray

from badappltex.formulas import ArithmeticFamily, FormulaGenerationError
from badappltex.masking import SilhouetteConfig, SilhouetteExtractor
from badappltex.placement import FormulaPlacer, PlacementConfig, compose_on_white
from badappltex.rendering import FormulaRenderError, MathTextRasterizer, RenderStyle


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from error

    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from error
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def _nonnegative_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from error
    if number < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return number


def _threshold(value: str) -> int:
    number = _nonnegative_integer(value)
    if number > 255:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 255")
    return number


class PreviewError(RuntimeError):
    """Raised when a requested CLI preview cannot be produced."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="badappltex",
        description=(
            "Preview one verified formula, or fill a source-image silhouette "
            "with verified formulas."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        metavar="IMAGE",
        help="use this image's dark regions as a formula-placement silhouette",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        metavar="PNG",
        help="write the black-on-white preview to this PNG file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed used to generate the formula (default: 0)",
    )
    parser.add_argument(
        "--font-size",
        type=_positive_float,
        default=16.0,
        metavar="POINTS",
        help="MathText font size in points (default: 16)",
    )
    parser.add_argument(
        "--dpi",
        type=_positive_float,
        default=100.0,
        help="rasterization resolution (default: 100)",
    )
    parser.add_argument(
        "--formula-count",
        type=_positive_integer,
        default=250,
        metavar="COUNT",
        help="formulas generated in source-image mode (default: 250)",
    )
    parser.add_argument(
        "--threshold",
        type=_threshold,
        default=127,
        metavar="0..255",
        help="maximum grayscale value treated as dark (default: 127)",
    )
    parser.add_argument(
        "--gap",
        type=_nonnegative_integer,
        default=1,
        metavar="PIXELS",
        help="minimum formula-ink separation (default: 1)",
    )
    parser.add_argument(
        "--placement-attempts",
        type=_positive_integer,
        default=256,
        metavar="COUNT",
        help="random positions tried per formula (default: 256)",
    )
    return parser


def _load_source(path: Path) -> NDArray[np.uint8]:
    if not path.is_file():
        raise PreviewError(f"input image does not exist: {path}")

    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise PreviewError(f"input image could not be decoded: {path}")
    return cast(NDArray[np.uint8], frame)


def _save_png(pixels: NDArray[np.uint8], output: Path) -> None:
    Image.fromarray(pixels).save(output, format="PNG")


def _formula_preview(args: argparse.Namespace) -> None:
    formula = ArithmeticFamily().generate(random.Random(args.seed))
    rendered = MathTextRasterizer().render(
        formula,
        RenderStyle(font_size_pt=args.font_size, dpi=args.dpi),
    )
    _save_png(np.subtract(255, rendered.alpha, dtype=np.uint8), args.output)

    print(f"formula: {formula.latex}")
    print(f"image: {rendered.width}x{rendered.height} -> {args.output}")


def _image_preview(args: argparse.Namespace) -> None:
    source = _load_source(args.input)
    silhouette = SilhouetteExtractor().extract(
        source,
        SilhouetteConfig(threshold=args.threshold),
    )

    rng = random.Random(args.seed)
    family = ArithmeticFamily()
    rasterizer = MathTextRasterizer()
    style = RenderStyle(font_size_pt=args.font_size, dpi=args.dpi)
    formulas = [
        rasterizer.render(family.generate(rng), style)
        for _ in range(args.formula_count)
    ]
    result = FormulaPlacer().place(
        silhouette,
        formulas,
        rng,
        PlacementConfig(
            attempts_per_formula=args.placement_attempts,
            gap_px=args.gap,
        ),
    )
    _save_png(compose_on_white(result), args.output)

    print(
        f"source: {silhouette.width}x{silhouette.height}, "
        f"foreground={silhouette.foreground_fraction:.2%}"
    )
    print(
        f"formulas: placed={len(result.placements)}, "
        f"rejected={result.rejected_formulas}"
    )
    print(
        f"coverage: core={result.core_coverage_fraction:.2%}, "
        f"support={result.support_coverage_fraction:.2%}, "
        f"outside={result.outside_ink_pixels}"
    )
    print(f"image: {result.width}x{result.height} -> {args.output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.input is None:
            _formula_preview(args)
        else:
            _image_preview(args)
    except (
        FormulaGenerationError,
        FormulaRenderError,
        OSError,
        PreviewError,
    ) as error:
        parser.error(str(error))

    return 0
