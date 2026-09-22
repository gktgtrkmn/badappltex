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

Convert an OpenCV frame into a placement silhouette:

```python
from badappltex.masking import SilhouetteConfig, SilhouetteExtractor

silhouette = SilhouetteExtractor().extract(
    frame,
    SilhouetteConfig(threshold=127, blur_radius_px=1),
)
print(silhouette.foreground_fraction)
```

Boolean `True` pixels represent regions where formula ink may be placed.
Three- and four-channel frames are interpreted as OpenCV-style BGR and BGRA.

Place pre-rendered formulas inside a silhouette and compose a white frame:

```python
import random

from badappltex.placement import FormulaPlacer, PlacementConfig, compose_on_white

result = FormulaPlacer().place(
    silhouette,
    rendered_formulas,
    random.Random(42),
    PlacementConfig(attempts_per_formula=128, gap_px=1),
)
frame = compose_on_white(result)
print(result.core_coverage_fraction, result.outside_ink_pixels)
```

Placement checks every nontransparent glyph pixel against the silhouette and
keeps configurable spacing between formula ink masks.

Generate a single black-on-white formula preview from the command line:

```console
uv run python -m badappltex --output preview.png --seed 42 --font-size 18
```

Fill the dark regions of a source image with formulas:

```console
uv run python -m badappltex \
  --input source.png \
  --output formula-silhouette.png \
  --seed 42 \
  --font-size 8 \
  --formula-count 300
```

Image mode also accepts `--threshold`, `--gap`, and `--placement-attempts` for
the controls already implemented by the core pipeline. A packaged
console-script entry point can be added when the CLI grows beyond this
development preview.

A ready-to-use test silhouette is included at
`examples/test-silhouette.png`:

```console
uv run python -m badappltex \
  --input examples/test-silhouette.png \
  --output preview.png \
  --seed 42 \
  --font-size 8 \
  --formula-count 350 \
  --placement-attempts 400
```

Run the tests with:

```console
uv run python -m unittest discover -s tests -v
```
