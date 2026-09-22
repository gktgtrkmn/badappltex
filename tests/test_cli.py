from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from badappltex.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_parser_defaults_are_intentionally_small(self) -> None:
        args = build_parser().parse_args(["--output", "preview.png"])

        self.assertEqual(args.output, Path("preview.png"))
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.font_size, 16.0)
        self.assertEqual(args.dpi, 100.0)
        self.assertIsNone(args.input)
        self.assertEqual(args.formula_count, 250)
        self.assertEqual(args.threshold, 127)
        self.assertEqual(args.gap, 1)
        self.assertEqual(args.placement_attempts, 256)

    def test_cli_writes_black_on_white_png_and_reports_formula(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview.png"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output",
                        str(output),
                        "--seed",
                        "42",
                        "--font-size",
                        "18",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                pixels = np.asarray(image)
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "L")
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)
                self.assertEqual(int(pixels.max()), 255)
                self.assertLess(int(pixels.min()), 255)

            report = stdout.getvalue()
            self.assertIn("formula:", report)
            self.assertIn("image:", report)
            self.assertIn(str(output), report)

    def test_same_seed_and_style_produce_identical_previews(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"

            with redirect_stdout(io.StringIO()):
                main(["-o", str(first), "--seed", "73"])
                main(["-o", str(second), "--seed", "73"])

            with Image.open(first) as first_image, Image.open(second) as second_image:
                np.testing.assert_array_equal(
                    np.asarray(first_image),
                    np.asarray(second_image),
                )

    def test_non_positive_render_values_are_parser_errors(self) -> None:
        for option in ("--font-size", "--dpi"):
            with self.subTest(option=option):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main(["-o", "unused.png", option, "0"])
                self.assertEqual(raised.exception.code, 2)

    def test_image_mode_preserves_dimensions_and_contains_all_ink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            output_path = Path(directory) / "result.png"
            source = np.full((120, 180), 255, dtype=np.uint8)
            source[10:110, 20:160] = 0
            Image.fromarray(source).save(source_path)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--input",
                        str(source_path),
                        "--output",
                        str(output_path),
                        "--seed",
                        "42",
                        "--font-size",
                        "7",
                        "--formula-count",
                        "80",
                        "--gap",
                        "0",
                        "--placement-attempts",
                        "500",
                    ]
                )

            self.assertEqual(exit_code, 0)
            with Image.open(output_path) as result_image:
                result = np.asarray(result_image)
                self.assertEqual(result.shape, source.shape)

            ink = result < 255
            self.assertTrue(np.any(ink))
            self.assertTrue(np.all(source[ink] <= 127))
            report = stdout.getvalue()
            self.assertIn("formulas: placed=", report)
            self.assertIn("coverage: core=", report)
            self.assertIn("outside=0", report)

    def test_missing_input_image_is_a_parser_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.png"
            output = Path(directory) / "unused.png"

            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["--input", str(missing), "--output", str(output)])

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_image_mode_integer_options_are_validated(self) -> None:
        invalid_options = (
            ("--formula-count", "0"),
            ("--threshold", "256"),
            ("--gap", "-1"),
            ("--placement-attempts", "0"),
        )

        for option, value in invalid_options:
            with self.subTest(option=option):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main(["-o", "unused.png", option, value])
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
