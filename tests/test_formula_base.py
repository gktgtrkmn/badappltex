from __future__ import annotations

import random
import unittest

import sympy as sp

from badappltex.formulas import Formula, FormulaFamily, FormulaGenerationError


class _NeverValidFamily(FormulaFamily):
    name = "never-valid"

    def _generate_candidate(self, rng: random.Random) -> Formula:
        del rng
        return Formula(
            lhs=sp.Integer(1),
            rhs=sp.Integer(2),
            latex="1 = 2",
            family=self.name,
            complexity=0,
        )

    def verify(self, formula: Formula) -> bool:
        del formula
        return False


class FormulaFamilyTests(unittest.TestCase):
    def test_generate_rejects_non_positive_attempt_count(self) -> None:
        with self.assertRaises(ValueError):
            _NeverValidFamily().generate(random.Random(0), max_attempts=0)

    def test_generate_raises_after_failed_attempts(self) -> None:
        with self.assertRaises(FormulaGenerationError):
            _NeverValidFamily().generate(random.Random(0), max_attempts=3)


if __name__ == "__main__":
    unittest.main()
