"""Place rasterized formulas inside a boolean silhouette."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from badappltex.masking import Silhouette
from badappltex.rendering import RenderedFormula


def _validate_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def _immutable_uint8(array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    buffer = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(array.shape)


def _immutable_bool(array: NDArray[np.bool_]) -> NDArray[np.bool_]:
    buffer = np.ascontiguousarray(array, dtype=np.bool_).tobytes()
    return np.frombuffer(buffer, dtype=np.bool_).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class PlacementConfig:
    """Controls randomized candidate search and glyph separation."""

    attempts_per_formula: int = 128
    gap_px: int = 1
    largest_first: bool = True

    def __post_init__(self) -> None:
        _validate_nonnegative_integer("attempts_per_formula", self.attempts_per_formula)
        if self.attempts_per_formula == 0:
            raise ValueError("attempts_per_formula must be at least 1")
        _validate_nonnegative_integer("gap_px", self.gap_px)
        if not isinstance(self.largest_first, bool):
            raise TypeError("largest_first must be a boolean")


@dataclass(frozen=True, slots=True)
class PlacedFormula:
    """A rasterized formula positioned in frame coordinates."""

    rendered: RenderedFormula
    x: int
    y: int

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (
            self.x,
            self.y,
            self.x + self.rendered.width,
            self.y + self.rendered.height,
        )


@dataclass(frozen=True, slots=True)
class PlacementResult:
    """Immutable placement canvases and quality metrics for one frame."""

    placements: tuple[PlacedFormula, ...]
    alpha: NDArray[np.uint8]
    support_mask: NDArray[np.bool_]
    core_mask: NDArray[np.bool_]
    rejected_formulas: int
    target_foreground_pixels: int
    outside_ink_pixels: int

    @property
    def width(self) -> int:
        return int(self.alpha.shape[1])

    @property
    def height(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def support_coverage_fraction(self) -> float:
        if self.target_foreground_pixels == 0:
            return 0.0
        return int(np.count_nonzero(self.support_mask)) / self.target_foreground_pixels

    @property
    def core_coverage_fraction(self) -> float:
        if self.target_foreground_pixels == 0:
            return 0.0
        return int(np.count_nonzero(self.core_mask)) / self.target_foreground_pixels


class FormulaPlacer:
    """Pack formula glyphs into a silhouette using seeded random candidates."""

    def place(
        self,
        silhouette: Silhouette,
        formulas: Iterable[RenderedFormula],
        rng: random.Random,
        config: PlacementConfig | None = None,
    ) -> PlacementResult:
        active_config = config or PlacementConfig()
        self._validate_silhouette(silhouette)

        candidates = tuple(formulas)
        for rendered in candidates:
            self._validate_rendered(rendered)

        if active_config.largest_first:
            candidates = tuple(
                sorted(
                    candidates,
                    key=lambda rendered: rendered.width * rendered.height,
                    reverse=True,
                )
            )

        height, width = silhouette.mask.shape
        alpha = np.zeros((height, width), dtype=np.uint8)
        support_canvas = np.zeros((height, width), dtype=np.bool_)
        core_canvas = np.zeros((height, width), dtype=np.bool_)
        blocked = np.zeros((height, width), dtype=np.bool_)
        placements: list[PlacedFormula] = []

        for rendered in candidates:
            position = self._find_position(
                silhouette.mask,
                blocked,
                rendered,
                rng,
                active_config.attempts_per_formula,
            )
            if position is None:
                continue

            x, y = position
            self._composite_masks(
                alpha,
                support_canvas,
                core_canvas,
                rendered,
                x,
                y,
            )
            spaced_support = self._spaced_support(
                rendered.support_mask,
                active_config.gap_px,
            )
            self._mark_blocked(
                blocked,
                spaced_support,
                x - active_config.gap_px,
                y - active_config.gap_px,
            )
            placements.append(PlacedFormula(rendered=rendered, x=x, y=y))

        outside_ink = int(
            np.count_nonzero(support_canvas & ~silhouette.mask)
        )
        return PlacementResult(
            placements=tuple(placements),
            alpha=_immutable_uint8(alpha),
            support_mask=_immutable_bool(support_canvas),
            core_mask=_immutable_bool(core_canvas),
            rejected_formulas=len(candidates) - len(placements),
            target_foreground_pixels=silhouette.foreground_pixels,
            outside_ink_pixels=outside_ink,
        )

    @staticmethod
    def _validate_silhouette(silhouette: Silhouette) -> None:
        if silhouette.mask.ndim != 2 or silhouette.mask.size == 0:
            raise ValueError("silhouette mask must be a nonempty 2D array")
        if silhouette.mask.dtype != np.bool_:
            raise TypeError("silhouette mask must have boolean pixel data")

    @staticmethod
    def _validate_rendered(rendered: RenderedFormula) -> None:
        shape = rendered.alpha.shape
        if rendered.alpha.ndim != 2 or rendered.alpha.size == 0:
            raise ValueError("rendered formula alpha must be a nonempty 2D array")
        if rendered.alpha.dtype != np.uint8:
            raise TypeError("rendered formula alpha must have uint8 pixel data")
        if rendered.support_mask.shape != shape or rendered.core_mask.shape != shape:
            raise ValueError("rendered formula arrays must have matching shapes")
        if rendered.support_mask.dtype != np.bool_:
            raise TypeError("rendered formula support mask must be boolean")
        if rendered.core_mask.dtype != np.bool_:
            raise TypeError("rendered formula core mask must be boolean")
        if not np.any(rendered.support_mask):
            raise ValueError("rendered formula support mask cannot be empty")
        if not np.array_equal(rendered.support_mask, rendered.alpha > 0):
            raise ValueError(
                "rendered formula support mask must match nonzero alpha pixels"
            )
        if np.any(rendered.core_mask & ~rendered.support_mask):
            raise ValueError("rendered formula core mask must be inside support")

    def _find_position(
        self,
        target: NDArray[np.bool_],
        blocked: NDArray[np.bool_],
        rendered: RenderedFormula,
        rng: random.Random,
        attempts: int,
    ) -> tuple[int, int] | None:
        frame_height, frame_width = target.shape
        if rendered.width > frame_width or rendered.height > frame_height:
            return None

        positions_x = frame_width - rendered.width + 1
        positions_y = frame_height - rendered.height + 1
        for x, y in self._candidate_positions(
            positions_x,
            positions_y,
            attempts,
            rng,
        ):
            target_region = target[y : y + rendered.height, x : x + rendered.width]
            if not np.all(target_region[rendered.support_mask]):
                continue

            blocked_region = blocked[y : y + rendered.height, x : x + rendered.width]
            if np.any(blocked_region[rendered.support_mask]):
                continue
            return x, y

        return None

    @staticmethod
    def _candidate_positions(
        positions_x: int,
        positions_y: int,
        attempts: int,
        rng: random.Random,
    ) -> Iterator[tuple[int, int]]:
        total = positions_x * positions_y
        count = min(attempts, total)

        indices = rng.sample(range(total), k=count)

        for index in indices:
            yield index % positions_x, index // positions_x

    @staticmethod
    def _composite_masks(
        alpha: NDArray[np.uint8],
        support_canvas: NDArray[np.bool_],
        core_canvas: NDArray[np.bool_],
        rendered: RenderedFormula,
        x: int,
        y: int,
    ) -> None:
        rows = slice(y, y + rendered.height)
        columns = slice(x, x + rendered.width)
        np.maximum(alpha[rows, columns], rendered.alpha, out=alpha[rows, columns])
        support_canvas[rows, columns] |= rendered.support_mask
        core_canvas[rows, columns] |= rendered.core_mask

    @staticmethod
    def _spaced_support(
        support: NDArray[np.bool_],
        gap_px: int,
    ) -> NDArray[np.bool_]:
        if gap_px == 0:
            return support

        padded = np.pad(support, gap_px, mode="constant", constant_values=False)
        size = 2 * gap_px + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        dilated = cv2.dilate(padded.astype(np.uint8), kernel)
        return cast(NDArray[np.bool_], dilated > 0)

    @staticmethod
    def _mark_blocked(
        blocked: NDArray[np.bool_],
        spaced_support: NDArray[np.bool_],
        x: int,
        y: int,
    ) -> None:
        frame_height, frame_width = blocked.shape
        local_height, local_width = spaced_support.shape
        frame_left = max(0, x)
        frame_top = max(0, y)
        frame_right = min(frame_width, x + local_width)
        frame_bottom = min(frame_height, y + local_height)

        local_left = frame_left - x
        local_top = frame_top - y
        local_right = local_left + (frame_right - frame_left)
        local_bottom = local_top + (frame_bottom - frame_top)
        blocked[frame_top:frame_bottom, frame_left:frame_right] |= spaced_support[
            local_top:local_bottom,
            local_left:local_right,
        ]


def compose_on_white(result: PlacementResult) -> NDArray[np.uint8]:
    """Return an immutable grayscale frame with black antialiased formula ink."""

    image = np.subtract(255, result.alpha, dtype=np.uint8)
    return _immutable_uint8(image)
