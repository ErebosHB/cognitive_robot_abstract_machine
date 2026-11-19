# Solid avocado demos

Four demonstrations in MuJoCo with an avocado of three solid parts, skin, flesh and pit
(`coraplex/resources/objects/solid_avocado`), on a cutting board. A knife cuts it exactly
where the blade goes through, a spoon scoops it exactly where the bowl goes, and the pit
is cut by neither. Every demo checks what it achieves; a failed check raises one of the
exceptions in `exceptions.py`. With a viewer, the window closes on its own 3 s after the
last step.

```bash
python cutting_demo.py        # halve the avocado, take the pit out, cut the halves
python skin_peeling_demo.py   # score the skin into strips and peel them off
python slicing_demo.py        # halve the avocado and slice both halves
python scooping_demo.py       # halve the avocado and take a scoop out of one half
python cutting_demo.py --headless           # without a viewer
python cutting_demo.py --speed 0.5          # the viewer at half speed
python cutting_demo.py --seed 9             # the random pit of seed 9 again
python test_cutting_demo.py                 # headless, as in CI; likewise the others
```

## The random pit

Before the scene is built, `random_pit.py` gives the avocado a pit of random size, place
and tilt, and shapes the cavity in the flesh to fit it. Every run prints the seed its pit
came from; `--seed` repeats a pit. A pit that would come too close to the skin is drawn
again.

## Cutting the free avocado (`cutting_demo.py`)

The avocado lies on the board. The knife halves it round the pit, the upper half is laid
aside cut face up and the pit lifted out, and each half is cut once more across. The
pieces are laid apart on the board.

## The held avocado

In the other three demos the avocado lies on its side on two thin poles standing on the
board, their tips stuck into its underside, clear of the pit and of every path of the
knife. A piece reaching into a pole is fixed in place; a piece cut off clear of the
poles is free.

### Peeling (`skin_peeling_demo.py`)

The knife scores the skin lengthwise into eight strips, cutting through the skin only.
Each strip is cut across into segments joined by hinges, comes off the flesh from its end
at the stem, straightening as it comes, and is laid flat along the board, where it rests
as a strip that bends. The strip the poles pierce stays on.

### Slicing (`slicing_demo.py`)

The knife halves the avocado round the pit; the upper half is laid aside and the pit
taken out. The half on the poles is cut crosswise into slices, each laid into a row on
the board as soon as it has fallen. Then the upper half is pushed onto the poles and
sliced the same way.

### Scooping (`scooping_demo.py`)

After the same halving, a spoon is led down into the half on the poles beside the pit's
hollow and up again. What its bowl held on that way comes away on it once the bowl,
having been full, comes out of the flesh, and is laid on the board; the rest stays in
the skin. The bowl slides along skin and pit in its way, and where they would stop it,
the spoon is drawn back out the way it came.

## How the avocado is cut

`cutting.py` cuts the meshes of the parts with `manifold3d` and compiles the scene again
with `MjSpec.recompile`, as MuJoCo cannot split a body while it simulates. Each piece is
a body of its own that carries on from the pose and velocity of the body it was cut
from, colliding through convex shapes that stay out of the other pieces. A knife cuts a
piece once its cutting edge has gone past the whole section in the blade's plane.
`knife_strokes.py` holds the strokes the demos make: straight down, round the pit, and
scoring the skin.

All four demos are in the CI matrix of `examples_and_demos.yml`, and
`test/coraplex_test/test_avocado_cutting.py` covers the cutting.
