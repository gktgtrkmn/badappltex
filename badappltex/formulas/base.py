"""Shared contracts for formula families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import Random

import sympy as sp


@dataclass(frozen=True, slots=True)
class Formula:
    """A display-ready mathematical statement and its provenance."""

    lhs: sp.Expr
    rhs: sp.Expr
    latex: str
    family: str
    complexity: int
    assumptions: tuple[str, ...] = ()


class FormulaGenerationError(RuntimeError):
    """Raised when a family cannot produce a valid formula in time."""


class FormulaFamily(ABC):
    """Generate and verify one mathematical family.

    Subclasses construct candidates and provide family-specific verification.
    The public :meth:`generate` method is the verification gate: a caller never
    receives a candidate that failed verification or quality checks.
    """

    name: str

    def generate(self, rng: Random, *, max_attempts: int = 100) -> Formula:
        """Return a verified formula using randomness supplied by ``rng``."""

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        for _ in range(max_attempts):
            candidate = self._generate_candidate(rng)
            if self.verify(candidate) and self.is_acceptable(candidate):
                return candidate

        raise FormulaGenerationError(
            f"{self.name!r} could not generate an acceptable formula "
            f"in {max_attempts} attempts"
        )

    @abstractmethod
    def _generate_candidate(self, rng: Random) -> Formula:
        """Construct one candidate without assuming it is valid."""

    @abstractmethod
    def verify(self, formula: Formula) -> bool:
        """Return whether ``formula`` is mathematically valid for this family."""

    def is_acceptable(self, formula: Formula) -> bool:
        """Apply non-mathematical quality checks to a verified candidate."""

        return True
