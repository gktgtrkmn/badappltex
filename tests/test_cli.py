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


if __name__ == "__main__":
    unittest.main()
