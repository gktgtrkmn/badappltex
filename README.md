# badappltex

`badappltex` will render Bad Apple!! silhouettes from randomized mathematical
formulas. Formula generation is built from independently extensible families;
each family owns both its construction rules and its mathematical verifier.

The first family generates bounded, exact arithmetic equalities:

```python
import random

from badappltex.formulas import ArithmeticFamily

rng = random.Random(42)
family = ArithmeticFamily()

formula = family.generate(rng)
print(formula.latex)
```

`generate()` only returns formulas that pass the family's exact verifier and
quality limits. Supplying a seeded `random.Random` instance makes generation
reproducible without relying on global random state.

Run the tests with:

```console
uv run python -m unittest discover -s tests -v
```
