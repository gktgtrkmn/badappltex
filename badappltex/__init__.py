"""Tools for rendering verified mathematical formulas as video silhouettes."""

from .formulas import (
    ArithmeticConfig,
    ArithmeticFamily,
    Formula,
    FormulaFamily,
    FormulaGenerationError,
)
from .rendering import (
    FormulaRenderError,
    MathTextRasterizer,
    RenderedFormula,
    RenderStyle,
)
from .masking import Silhouette, SilhouetteConfig, SilhouetteExtractor
from .placement import (
    FormulaPlacer,
    PlacedFormula,
    PlacementConfig,
    PlacementResult,
    compose_on_white,
)

__all__ = [
    "ArithmeticConfig",
    "ArithmeticFamily",
    "Formula",
    "FormulaFamily",
    "FormulaGenerationError",
    "FormulaRenderError",
    "MathTextRasterizer",
    "RenderedFormula",
    "RenderStyle",
    "Silhouette",
    "SilhouetteConfig",
    "SilhouetteExtractor",
    "FormulaPlacer",
    "PlacedFormula",
    "PlacementConfig",
    "PlacementResult",
    "compose_on_white",
]
