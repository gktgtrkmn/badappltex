"""Generation of exact, bounded arithmetic equalities."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal, cast

import sympy as sp

from .base import Formula, FormulaFamily

ArithmeticOperation = Literal["add", "subtract", "multiply", "divide"]
_OPERATIONS: tuple[ArithmeticOperation, ...] = (
    "add",
    "subtract",
    "multiply",
    "divide",
)


@dataclass(frozen=True, slots=True)
class ArithmeticConfig:
    """Bounds that keep generated equations readable and renderable."""

    minimum_leaf: int = 1
    maximum_leaf: int = 9
    minimum_depth: int = 1
    maximum_depth: int = 3
    maximum_numerator: int = 10_000
    maximum_denominator: int = 1_000
    maximum_latex_length: int = 120

    def __post_init__(self) -> None:
        if self.minimum_leaf > self.maximum_leaf:
            raise ValueError("minimum_leaf cannot exceed maximum_leaf")
        if self.minimum_depth < 1:
            raise ValueError("minimum_depth must be at least 1")
        if self.minimum_depth > self.maximum_depth:
            raise ValueError("minimum_depth cannot exceed maximum_depth")
        if self.maximum_numerator < 1:
            raise ValueError("maximum_numerator must be positive")
        if self.maximum_denominator < 1:
            raise ValueError("maximum_denominator must be positive")
        if self.maximum_latex_length < 1:
            raise ValueError("maximum_latex_length must be positive")


@dataclass(frozen=True, slots=True)
class _ArithmeticNode:
    display: sp.Expr
    value: sp.Rational
    latex: str
    precedence: int


class ArithmeticFamily(FormulaFamily):
    """Create arithmetic expressions equal to exact integer or rational values."""

    name = "arithmetic"

    def __init__(self, config: ArithmeticConfig | None = None) -> None:
        self.config = config or ArithmeticConfig()

    def _generate_candidate(self, rng: Random) -> Formula:
        depth = rng.randint(self.config.minimum_depth, self.config.maximum_depth)
        node = self._generate_node(rng, depth)
        rhs = node.value

        return Formula(
            lhs=node.display,
            rhs=rhs,
            latex=rf"{node.latex} = {sp.latex(rhs)}",
            family=self.name,
            complexity=int(sp.count_ops(node.display)),
        )

    def _generate_node(self, rng: Random, depth: int) -> _ArithmeticNode:
        if depth == 0:
            value = cast(
                sp.Rational,
                sp.Rational(
                    rng.randint(self.config.minimum_leaf, self.config.maximum_leaf)
                ),
            )
            latex = sp.latex(value)
            if value < 0:
                latex = rf"\left({latex}\right)"
            return _ArithmeticNode(
                display=value,
                value=value,
                latex=latex,
                precedence=3,
            )

        operation = rng.choice(_OPERATIONS)
        left = self._generate_node(rng, depth - 1)
        right = self._generate_node(rng, depth - 1)

        if operation == "add":
            display = sp.Add(left.display, right.display, evaluate=False)
            value = cast(sp.Rational, left.value + right.value)
            latex = rf"{left.latex} + {right.latex}"
            precedence = 1
        elif operation == "subtract":
            negative_right = sp.Mul(-1, right.display, evaluate=False)
            display = sp.Add(left.display, negative_right, evaluate=False)
            value = cast(sp.Rational, left.value - right.value)
            latex = rf"{left.latex} - {self._group(right, 1, include_equal=True)}"
            precedence = 1
        elif operation == "multiply":
            display = sp.Mul(left.display, right.display, evaluate=False)
            value = cast(sp.Rational, left.value * right.value)
            latex = (
                rf"{self._group(left, 2)} \times {self._group(right, 2)}"
            )
            precedence = 2
        else:
            right = self._ensure_nonzero_denominator(rng, right, depth - 1)
            reciprocal = sp.Pow(right.display, -1, evaluate=False)
            display = sp.Mul(left.display, reciprocal, evaluate=False)
            value = cast(sp.Rational, left.value / right.value)
            latex = rf"\frac{{{left.latex}}}{{{right.latex}}}"
            precedence = 2

        return _ArithmeticNode(
            display=display,
            value=value,
            latex=latex,
            precedence=precedence,
        )

    @staticmethod
    def _group(
        node: _ArithmeticNode,
        minimum_precedence: int,
        *,
        include_equal: bool = False,
    ) -> str:
        needs_grouping = node.precedence < minimum_precedence or (
            include_equal and node.precedence == minimum_precedence
        )
        if needs_grouping:
            return rf"\left({node.latex}\right)"
        return node.latex

    def _ensure_nonzero_denominator(
        self,
        rng: Random,
        denominator: _ArithmeticNode,
        depth: int,
    ) -> _ArithmeticNode:
        for _ in range(20):
            if denominator.value != 0:
                return denominator
            denominator = self._generate_node(rng, depth)

        one = cast(sp.Rational, sp.Rational(1))
        return _ArithmeticNode(display=one, value=one, latex="1", precedence=3)

    def verify(self, formula: Formula) -> bool:
        """Verify the displayed equality with exact symbolic arithmetic."""

        if formula.family != self.name:
            return False
        if formula.lhs.free_symbols or formula.rhs.free_symbols:
            return False
        if formula.lhs.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return False
        if formula.rhs.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return False

        try:
            difference = sp.cancel(sp.together(formula.lhs - formula.rhs))
        except (TypeError, ValueError, ZeroDivisionError):
            return False

        return difference == 0

    def is_acceptable(self, formula: Formula) -> bool:
        """Reject correct but visually unhelpful or excessively large results."""

        if formula.lhs == formula.rhs:
            return False
        if sp.latex(formula.lhs) == sp.latex(formula.rhs):
            return False
        if len(formula.latex) > self.config.maximum_latex_length:
            return False
        if not formula.rhs.is_Rational:
            return False

        numerator, denominator = formula.rhs.as_numer_denom()
        return (
            abs(int(numerator)) <= self.config.maximum_numerator
            and int(denominator) <= self.config.maximum_denominator
        )
