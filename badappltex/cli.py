"""Minimal command-line preview for the implemented formula pipeline."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

from PIL import Image

from badappltex.formulas import ArithmeticFamily, FormulaGenerationError
from badappltex.rendering import FormulaRenderError, MathTextRasterizer, RenderStyle


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from error

    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="badappltex",
        description="Generate and rasterize one verified arithmetic formula.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        formula = ArithmeticFamily().generate(random.Random(args.seed))
        rendered = MathTextRasterizer().render(
            formula,
            RenderStyle(font_size_pt=args.font_size, dpi=args.dpi),
        )
        preview = Image.fromarray(255 - rendered.alpha)
        preview.save(args.output, format="PNG")
    except (FormulaGenerationError, FormulaRenderError, OSError) as error:
        parser.error(str(error))

    print(f"formula: {formula.latex}")
    print(f"image: {rendered.width}x{rendered.height} -> {args.output}")
    return 0
