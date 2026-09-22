from __future__ import annotations

import unittest

import numpy as np

from badappltex.masking import SilhouetteConfig, SilhouetteExtractor


class SilhouetteConfigTests(unittest.TestCase):
    def test_invalid_values_are_rejected(self) -> None:
        for threshold in (-1, 256):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    SilhouetteConfig(threshold=threshold)

        with self.assertRaises(TypeError):
            SilhouetteConfig(threshold=True)
        with self.assertRaises(TypeError):
            SilhouetteConfig(dark_is_foreground=1)  # type: ignore[arg-type]

        for field in (
            "blur_radius_px",
            "opening_radius_px",
            "closing_radius_px",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    SilhouetteConfig(**{field: -1})  # type: ignore[arg-type]


class SilhouetteExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = SilhouetteExtractor()

    def test_dark_threshold_is_inclusive(self) -> None:
        frame = np.array([[0, 127, 128, 255]], dtype=np.uint8)
        silhouette = self.extractor.extract(
            frame,
            SilhouetteConfig(threshold=127),
        )

        expected = np.array([[True, True, False, False]])
        np.testing.assert_array_equal(silhouette.mask, expected)

    def test_foreground_can_be_inverted(self) -> None:
        frame = np.array([[0, 127, 128, 255]], dtype=np.uint8)
        silhouette = self.extractor.extract(
            frame,
            SilhouetteConfig(threshold=127, dark_is_foreground=False),
        )

        expected = np.array([[False, False, True, True]])
        np.testing.assert_array_equal(silhouette.mask, expected)

    def test_grayscale_bgr_bgra_and_single_channel_frames_match(self) -> None:
        grayscale = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        bgr = np.repeat(grayscale[:, :, np.newaxis], 3, axis=2)
        bgra = np.concatenate(
            [bgr, np.full((2, 2, 1), 127, dtype=np.uint8)],
            axis=2,
        )
        single_channel = grayscale[:, :, np.newaxis]

        expected = self.extractor.extract(grayscale).mask
        for frame in (bgr, bgra, single_channel):
            with self.subTest(shape=frame.shape):
                actual = self.extractor.extract(frame).mask
                np.testing.assert_array_equal(actual, expected)

    def test_blur_can_remove_a_single_dark_impulse(self) -> None:
        frame = np.full((5, 5), 255, dtype=np.uint8)
        frame[2, 2] = 0

        sharp = self.extractor.extract(frame)
        blurred = self.extractor.extract(
            frame,
            SilhouetteConfig(blur_radius_px=1),
        )

        self.assertTrue(sharp.mask[2, 2])
        self.assertFalse(np.any(blurred.mask))

    def test_opening_removes_isolated_foreground_pixel(self) -> None:
        frame = np.full((7, 7), 255, dtype=np.uint8)
        frame[3, 3] = 0

        silhouette = self.extractor.extract(
            frame,
            SilhouetteConfig(opening_radius_px=1),
        )

        self.assertFalse(np.any(silhouette.mask))

    def test_closing_fills_isolated_background_hole(self) -> None:
        frame = np.zeros((7, 7), dtype=np.uint8)
        frame[3, 3] = 255

        silhouette = self.extractor.extract(
            frame,
            SilhouetteConfig(closing_radius_px=1),
        )

        self.assertTrue(np.all(silhouette.mask))

    def test_result_is_immutable_and_reports_metrics(self) -> None:
        frame = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        silhouette = self.extractor.extract(frame)

        self.assertEqual(silhouette.width, 2)
        self.assertEqual(silhouette.height, 2)
        self.assertEqual(silhouette.foreground_pixels, 2)
        self.assertEqual(silhouette.foreground_fraction, 0.5)
        with self.assertRaises(ValueError):
            silhouette.mask[0, 0] = False
        with self.assertRaises(ValueError):
            silhouette.mask.setflags(write=True)

    def test_invalid_frames_are_rejected(self) -> None:
        invalid_frames = (
            np.array([], dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2, 2), dtype=np.uint8),
            np.zeros((1, 2, 2, 1), dtype=np.uint8),
        )

        for frame in invalid_frames:
            with self.subTest(shape=frame.shape, dtype=frame.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    self.extractor.extract(frame)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
