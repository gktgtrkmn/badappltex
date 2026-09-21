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

Render a formula into immutable alpha and placement masks with Matplotlib
MathText:

```python
from badappltex.rendering import MathTextRasterizer, RenderStyle

rasterizer = MathTextRasterizer()
rendered = rasterizer.render(formula, RenderStyle(font_size_pt=16))

print(rendered.alpha.shape)
print(rendered.support_mask.shape)
```

The support mask contains every antialiased glyph pixel and is intended for
strict silhouette containment. The core mask contains pixels at or above the
configured alpha threshold and is intended for coverage measurements.

Generate a black-on-white PNG preview from the command line:

```console
uv run python -m badappltex --output preview.png --seed 42 --font-size 18
```

This small CLI intentionally exposes only the formula and rasterization
controls that exist today; placement and video options will be added with those
stages. A packaged console-script entry point can be added when the CLI grows
beyond this development preview.

Run the tests with:

```console
uv run python -m unittest discover -s tests -v
```
