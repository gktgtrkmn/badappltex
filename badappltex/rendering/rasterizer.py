"""Render formula LaTeX into immutable glyph masks with Matplotlib MathText."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite
from threading import RLock

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from numpy.typing import NDArray

from badappltex.formulas import Formula


class FormulaRenderError(RuntimeError):
    """Raised when a formula cannot be converted into a nonempty bitmap."""


@dataclass(frozen=True, slots=True)
class RenderStyle:
    """Rendering inputs that affect the generated pixels."""

    font_size_pt: float
    dpi: float = 100.0
    padding_px: int = 1
    core_threshold: int = 128

    def __post_init__(self) -> None:
        if not isfinite(self.font_size_pt) or self.font_size_pt <= 0:
            raise ValueError("font_size_pt must be finite and positive")
        if not isfinite(self.dpi) or self.dpi <= 0:
            raise ValueError("dpi must be finite and positive")
        if self.padding_px < 0:
            raise ValueError("padding_px cannot be negative")
        if not 1 <= self.core_threshold <= 255:
            raise ValueError("core_threshold must be between 1 and 255")


@dataclass(frozen=True, slots=True)
class RenderedFormula:
    """A tightly cropped formula bitmap and masks derived from its alpha."""

    latex: str
    alpha: NDArray[np.uint8]
    support_mask: NDArray[np.bool_]
    core_mask: NDArray[np.bool_]
    baseline_depth: float

    @property
    def width(self) -> int:
        return int(self.alpha.shape[1])

    @property
    def height(self) -> int:
        return int(self.alpha.shape[0])


_CacheKey = tuple[str, RenderStyle]


def _immutable_uint8(array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    buffer = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(array.shape)


def _immutable_bool(array: NDArray[np.bool_]) -> NDArray[np.bool_]:
    buffer = np.ascontiguousarray(array, dtype=np.bool_).tobytes()
    return np.frombuffer(buffer, dtype=np.bool_).reshape(array.shape)


class MathTextRasterizer:
    """Rasterize formulas with a small, immutable least-recently-used cache."""

    def __init__(self, *, max_cache_entries: int = 512) -> None:
        if max_cache_entries < 0:
            raise ValueError("max_cache_entries cannot be negative")

        self._max_cache_entries = max_cache_entries
        self._cache: OrderedDict[_CacheKey, RenderedFormula] = OrderedDict()
        self._parser = MathTextParser("agg")
        self._lock = RLock()

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def render(self, formula: Formula, style: RenderStyle) -> RenderedFormula:
        """Return an immutable raster for ``formula`` using ``style``."""

        key = (formula.latex, style)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

            rendered = self._render_uncached(formula.latex, style)
            if self._max_cache_entries > 0:
                self._cache[key] = rendered
                self._cache.move_to_end(key)
                while len(self._cache) > self._max_cache_entries:
                    self._cache.popitem(last=False)

            return rendered

    def _render_uncached(
        self,
        latex: str,
        style: RenderStyle,
    ) -> RenderedFormula:
        if not latex.strip():
            raise FormulaRenderError("formula LaTeX cannot be empty")
        if "$" in latex:
            raise FormulaRenderError(
                "formula LaTeX must not include MathText dollar delimiters"
            )

        math_text = f"${latex}$"
        font = FontProperties(size=style.font_size_pt)
        try:
            parsed = self._parser.parse(
                math_text,
                dpi=style.dpi,
                prop=font,
                antialiased=True,
            )
        except (TypeError, ValueError) as error:
            raise FormulaRenderError(
                f"MathText could not render formula {latex!r}"
            ) from error

        raw_alpha = np.array(parsed.image, dtype=np.uint8, copy=True)
        raw_support = raw_alpha > 0
        ink_y, ink_x = np.nonzero(raw_support)
        if ink_y.size == 0 or ink_x.size == 0:
            raise FormulaRenderError(
                f"MathText produced an empty bitmap for formula {latex!r}"
            )

        top = int(ink_y.min())
        bottom = int(ink_y.max()) + 1
        left = int(ink_x.min())
        right = int(ink_x.max()) + 1
        alpha = raw_alpha[top:bottom, left:right]

        if style.padding_px:
            alpha = np.pad(
                alpha,
                pad_width=style.padding_px,
                mode="constant",
                constant_values=0,
            )
        else:
            alpha = np.array(alpha, copy=True)

        support_mask = alpha > 0
        core_mask = alpha >= style.core_threshold

        # ``depth`` is measured upward from the bottom of MathText's original
        # bitmap. Account for cropped bottom rows and the new padding.
        removed_bottom_rows = raw_alpha.shape[0] - bottom
        baseline_depth = (
            float(parsed.depth)
            - float(removed_bottom_rows)
            + float(style.padding_px)
        )

        # Arrays backed by immutable ``bytes`` cannot have writeability turned
        # back on, which protects every user of a shared cache entry.
        alpha = _immutable_uint8(alpha)
        support_mask = _immutable_bool(support_mask)
        core_mask = _immutable_bool(core_mask)

        return RenderedFormula(
            latex=latex,
            alpha=alpha,
            support_mask=support_mask,
            core_mask=core_mask,
            baseline_depth=baseline_depth,
        )
