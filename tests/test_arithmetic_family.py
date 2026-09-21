from __future__ import annotations

import random
import unittest

import sympy as sp

from badappltex.formulas import ArithmeticConfig, ArithmeticFamily, Formula


class ArithmeticFamilyTests(unittest.TestCase):
    def test_generated_formulas_are_verified(self) -> None:
        family = ArithmeticFamily()
        rng = random.Random(20260921)

        for _ in range(250):
            formula = family.generate(rng)
            self.assertTrue(family.verify(formula), formula.latex)
            self.assertTrue(family.is_acceptable(formula), formula.latex)

    def test_generation_is_reproducible_for_a_seed(self) -> None:
        family = ArithmeticFamily()

        first_rng = random.Random(42)
        second_rng = random.Random(42)
        first = [family.generate(first_rng).latex for _ in range(20)]
        second = [family.generate(second_rng).latex for _ in range(20)]

        self.assertEqual(first, second)

    def test_nested_subtraction_is_visually_grouped(self) -> None:
        family = ArithmeticFamily(
            ArithmeticConfig(minimum_depth=3, maximum_depth=3)
        )
        formulas = [family.generate(random.Random(seed)) for seed in range(50)]

        # This pattern proves that a subtraction used as the right operand of
        # another subtraction remains grouped in the displayed equation.
        self.assertTrue(
            any(r"- \left(" in formula.latex for formula in formulas)
        )

    def test_result_respects_configured_bounds(self) -> None:
        config = ArithmeticConfig(
            maximum_numerator=20,
            maximum_denominator=5,
            maximum_latex_length=60,
        )
        family = ArithmeticFamily(config)
        rng = random.Random(7)

        for _ in range(100):
            formula = family.generate(rng)
            numerator, denominator = formula.rhs.as_numer_denom()
            self.assertLessEqual(abs(int(numerator)), 20)
            self.assertLessEqual(int(denominator), 5)
            self.assertLessEqual(len(formula.latex), 60)

    def test_false_equation_is_rejected(self) -> None:
        family = ArithmeticFamily()
        formula = Formula(
            lhs=sp.Add(2, 2, evaluate=False),
            rhs=sp.Integer(5),
            latex="2 + 2 = 5",
            family=family.name,
            complexity=1,
        )

        self.assertFalse(family.verify(formula))

    def test_symbolic_equation_is_outside_this_family(self) -> None:
        family = ArithmeticFamily()
        x = sp.Symbol("x")
        formula = Formula(
            lhs=x + 1,
            rhs=x + 1,
            latex="x + 1 = x + 1",
            family=family.name,
            complexity=1,
        )

        self.assertFalse(family.verify(formula))

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArithmeticConfig(minimum_depth=0)


if __name__ == "__main__":
    unittest.main()
