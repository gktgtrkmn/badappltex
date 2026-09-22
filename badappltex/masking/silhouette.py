"""Convert decoded video frames into immutable formula-placement masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def _validate_radius(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class SilhouetteConfig:
    """Controls deterministic conversion from a frame to a boolean mask."""

    threshold: int = 127
    dark_is_foreground: bool = True
    blur_radius_px: int = 0
    opening_radius_px: int = 0
    closing_radius_px: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, int):
            raise TypeError("threshold must be an integer")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if not isinstance(self.dark_is_foreground, bool):
            raise TypeError("dark_is_foreground must be a boolean")
        _validate_radius("blur_radius_px", self.blur_radius_px)
        _validate_radius("opening_radius_px", self.opening_radius_px)
        _validate_radius("closing_radius_px", self.closing_radius_px)


@dataclass(frozen=True, slots=True)
class Silhouette:
    """An immutable mask where true pixels may contain formula ink."""

    mask: NDArray[np.bool_]

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def foreground_pixels(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def foreground_fraction(self) -> float:
        return self.foreground_pixels / self.mask.size


class SilhouetteExtractor:
    """Extract formula-placement masks from OpenCV-style uint8 frames.

    Three- and four-channel inputs are interpreted as BGR and BGRA, matching
    frames returned by ``cv2.VideoCapture``.
    """

    def extract(
        self,
        frame: NDArray[np.uint8],
        config: SilhouetteConfig | None = None,
    ) -> Silhouette:
        active_config = config or SilhouetteConfig()
        grayscale = self._to_grayscale(frame)

        if active_config.blur_radius_px:
            kernel_size = 2 * active_config.blur_radius_px + 1
            grayscale = cv2.GaussianBlur(
                grayscale,
                (kernel_size, kernel_size),
                sigmaX=0,
                borderType=cv2.BORDER_REPLICATE,
            )

        if active_config.dark_is_foreground:
            mask = grayscale <= active_config.threshold
        else:
            mask = grayscale > active_config.threshold

        mask = self._apply_morphology(mask, active_config)
        frozen_mask = self._freeze(mask)
        return Silhouette(mask=frozen_mask)

    @staticmethod
    def _to_grayscale(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")
        if frame.dtype != np.uint8:
            raise TypeError("frame must have uint8 pixel data")
        if frame.size == 0:
            raise ValueError("frame cannot be empty")

        if frame.ndim == 2:
            return np.array(frame, dtype=np.uint8, copy=True)
        if frame.ndim != 3:
            raise ValueError("frame must have shape (H, W), (H, W, 1/3/4)")

        channels = frame.shape[2]
        if channels == 1:
            return np.array(frame[:, :, 0], dtype=np.uint8, copy=True)
        if channels == 3:
            return cast(
                NDArray[np.uint8],
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            )
        if channels == 4:
            return cast(
                NDArray[np.uint8],
                cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY),
            )
        raise ValueError("frame must have 1, 3, or 4 channels")

    @staticmethod
    def _apply_morphology(
        mask: NDArray[np.bool_],
        config: SilhouetteConfig,
    ) -> NDArray[np.bool_]:
        pixels = mask.astype(np.uint8) * 255

        if config.opening_radius_px:
            kernel = SilhouetteExtractor._ellipse_kernel(config.opening_radius_px)
            pixels = cv2.morphologyEx(pixels, cv2.MORPH_OPEN, kernel)
        if config.closing_radius_px:
            kernel = SilhouetteExtractor._ellipse_kernel(config.closing_radius_px)
            pixels = cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, kernel)

        return pixels > 0

    @staticmethod
    def _ellipse_kernel(radius: int) -> NDArray[np.uint8]:
        size = 2 * radius + 1
        return cast(
            NDArray[np.uint8],
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )

    @staticmethod
    def _freeze(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        buffer = np.ascontiguousarray(mask, dtype=np.bool_).tobytes()
        return np.frombuffer(buffer, dtype=np.bool_).reshape(mask.shape)
