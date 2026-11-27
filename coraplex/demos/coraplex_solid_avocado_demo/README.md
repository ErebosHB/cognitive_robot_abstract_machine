# Solid avocado demos

Four MuJoCo demonstrations of cutting, peeling, slicing and scooping an avocado that
behaves like a real one: it has three solid parts, skin, flesh and pit, a knife cuts it
exactly where the blade goes through, a spoon scoops exactly what its bowl holds, and
neither cuts the pit. Each piece cut off becomes a rigid body of its own that keeps
moving physically, and every piece is also mirrored into a `semantic_digital_twin`
`World`, so that planners and knowledge bases can reason about the pieces. The pieces a
demonstration ends with can be written to USD and simulated further in Isaac Sim.

| Demo                   | What happens                                                        |
|------------------------|---------------------------------------------------------------------|
| `cutting_demo.py`      | Halve the avocado round the pit, take the pit out, cut the halves.  |
| `skin_peeling_demo.py` | Score the skin into strips and peel them off as bendable strips.    |
| `slicing_demo.py`      | Halve the avocado and cut both halves into slices.                  |
| `scooping_demo.py`     | Halve the avocado and take one spoonful of flesh out of one half.   |

## Quick start

The demos need the CRAM workspace's Python environment, with `coraplex` and
`semantic_digital_twin` installed; that brings `mujoco`, `manifold3d` and `trimesh`.
The demo modules import each other by plain name, so run them from this folder:

```bash
cd coraplex/demos/coraplex_solid_avocado_demo
python cutting_demo.py
```

A MuJoCo viewer opens, plays the demonstration in real time and closes on its own 3 s
after the last step. Every demo takes the same options:

| Option       | Effect                                                               |
|--------------|----------------------------------------------------------------------|
| `--headless` | Run without a viewer, as fast as the machine allows.                 |
| `--speed X`  | Simulated seconds shown per second; `0.5` is half speed, `4` faster. |
| `--seed N`   | Use the random pit of seed `N` again; see below.                     |

At the end each demo prints what it achieved, for example:

```text
The pit came from seed 9.
Cut the avocado into pieces of 20, 120, 34, 85 cm³, every cut within 4 nm of the
blade's plane, and took the pit out as avocado_piece_5.
```

Every demo checks its own result. If something goes wrong, for example a cut is never
finished, pieces fly off or the simulation becomes unstable, it stops with one of the
exceptions in `exceptions.py`, which says what happened and what to try.
`test_<demo>.py` runs a demo headless, the way CI does.

## The demonstrations

### Cutting the free avocado (`cutting_demo.py`)

The avocado lies on the board. The knife halves it lengthwise, going round the pit. The
upper half is laid aside cut face up and the pit is lifted out. Then each half is cut
once more, and the pieces are laid apart on the board.

### The held avocado

In the other three demos, the avocado lies on its side on two thin poles standing on the
board. The poles' tips are stuck into its underside, clear of the pit and of every path
of the knife. A piece that a pole reaches into is held in place; a piece cut off clear
of the poles falls freely.

- **Peeling (`skin_peeling_demo.py`).** The knife scores the skin lengthwise into eight
  strips, cutting through the skin only. Each strip is cut across into segments joined
  by hinges, comes off the flesh from its stem end, straightens as it comes, and is laid
  flat on the board, where it rests as a strip that bends. The strip the poles pierce
  stays on.
- **Slicing (`slicing_demo.py`).** The knife halves the avocado and the pit is taken
  out. The half on the poles is cut crosswise into slices, and each slice is laid into a
  row on the board as soon as it has fallen. Then the other half is pushed onto the
  poles and sliced the same way.
- **Scooping (`scooping_demo.py`).** After the same halving, a spoon is led down into
  the half on the poles beside the pit's hollow and up again. The flesh its bowl held on
  the way comes out with it and is laid on the board; the rest stays in the skin. The
  bowl slides along the skin and the pit instead of cutting into them, and where they
  would stop it, the spoon backs out the way it came.

### A different pit every run

Real avocados differ, so each run gives the avocado a pit of random size (85–110 % of
the model's), position (up to 4 mm along the fruit and 2 mm across) and tilt (up to
about 20°). The cavity in the flesh is reshaped to fit the new pit. A pit that would
come closer than 1 mm to the skin is drawn again. Each run prints its seed, and `--seed`
repeats that pit; in code, pass `random_pit=None` to use the model's own pit.

## Using the pieces from your own code

The demonstrations are dataclasses. Build one, run it, and read the pieces from its
`semantic_digital_twin` world afterwards, or while it runs:

```python
from slicing_demo import AvocadoSlicingDemonstration

demonstration = AvocadoSlicingDemonstration(headless=True, pit_seed=9)
outcome = demonstration.run()

world = demonstration.fruit_world.world
for body in world.bodies:
    position = world.compute_forward_kinematics_np(world.root, body)[:3, 3]
    print(body.name.name, position.round(3))
```

The world, built by `fruit_world.py`, holds:

- a root body `world`, in the frame of the MuJoCo scene;
- one body per piece, named like its MuJoCo body (`avocado` before the first cut,
  `avocado_piece_<n>` after it). It hangs from the root by a free 6-DoF connection,
  has the piece's mass, centre of mass and inertia, and collides through the same
  convex shapes as the piece in MuJoCo;
- below each piece, one body per part it contains, `<piece>_skin`, `<piece>_flesh` and
  `<piece>_pit`, shaped and coloured like the part and weightless, as the piece carries
  the mass.

Each piece's body is annotated with what it is, using the annotations of
`semantic_digital_twin.semantic_annotations`: the whole avocado before the first cut as
an `Avocado` (a `Fruit`), every piece cut or taken out of it as an `AvocadoPiece`. Its
`peel`, `flesh` and `pit` fields hold the `Peel`, `Flesh` and `Pit` annotations of the
part bodies it has, and are `None` for a part it lacks. The world,
annotations included, can be stored in a database through the `semantic_digital_twin`
ORM, for KnowRob or anything else to query:

```python
from semantic_digital_twin.semantic_annotations.semantic_annotations import AvocadoPiece

for piece in world.get_semantic_annotations_by_type(AvocadoPiece):
    if piece.pit is not None:
        print(piece.root.name.name, "holds the pit")
```

When a cut is finished, the cut piece's bodies and annotations leave the world and the
new pieces' bodies and annotations join it. The poses follow the simulation every 0.05 s of simulated time
(`world_update_interval`) and once more at the end of a run. Pass `fruit_world=None` to
a demonstration to keep no world. The world's own collision detector is paused, as MuJoCo
checks the collisions; see the docstring of `FruitWorld.world` to resume it.

> **Note:** A piece's frame is the frame of the whole avocado it was cut from, carried
> along with the piece: its origin is where the avocado's base was, not the piece's
> centre. The meshes are given in that frame.

## Publishing the pieces on ROS 2

`publish_on_ros.py` runs a demonstration while it publishes the avocado's world on ROS 2,
for KnowRob or any other process to follow. It takes the demos' options:

```bash
source /opt/ros/jazzy/setup.bash
python publish_on_ros.py scooping --seed 9
```

| Published on                          | What                                                              |
|---------------------------------------|-------------------------------------------------------------------|
| `/semantic_digital_twin/world_sync`   | Every change: pieces and annotations added or removed, poses.     |
| `/semantic_digital_twin/fetch_world`  | A service handing out the whole world, to join it at any time.    |
| `tf`                                  | The frame of each piece.                                          |
| `tf_static`                           | The frame of each part, fixed to its piece.                       |

When the demonstration has ended, the script keeps publishing the pieces it ended with
until Enter is pressed. A process with `semantic_digital_twin` joins the world, and
from then on keeps it in step, like this:

```python
from semantic_digital_twin.adapters.ros.background_node import get_node, init_node
from semantic_digital_twin.adapters.ros.world_fetcher import join_shared_world
from semantic_digital_twin.semantic_annotations.semantic_annotations import AvocadoPiece

init_node("avocado_listener")
world = join_shared_world(get_node())
for piece in world.get_semantic_annotations_by_type(AvocadoPiece):
    print(piece.root.name.name, piece.pit is not None)
```

The meshes travel with the world, so the listener needs neither these demos nor their
files.

## Asking KnowRob about the pieces

`ask_knowrob.py` joins the world `publish_on_ros.py` publishes, keeps a
[KnowRob](https://github.com/knowrob/knowrob) knowledge base knowing what the pieces are
while they are cut, and answers the KnowRob queries typed in, one per line. Start the
publisher first, then, in a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
python ask_knowrob.py
```

```text
dul:hasPart(?piece, ?pit), rdf:type(?pit, sdt:'Pit')
  ?piece = https://github.com/cram2/.../world#avocado_piece_3, ?pit = https://github.com/cram2/.../world#avocado_piece_3_pit
1 answers.
```

The bridge, `KnowRobBridge` in `semantic_digital_twin.adapters.knowrob.bridge`, works
for any `semantic_digital_twin` world. It tells KnowRob about every semantic annotation
of the world, and forgets an annotation when it leaves the world:

| Statement                                  | Meaning                                                  |
|--------------------------------------------|----------------------------------------------------------|
| `rdf:type(world:avocado_piece_3, sdt:'AvocadoPiece')` | The annotation is of its class, and of each class above it (`sdt:'Food'`). |
| `rdfs:subClassOf(sdt:'AvocadoPiece', sdt:'Food')` | The classes of the annotations and the class above each.  |
| `dul:hasPart(world:avocado_piece_3, world:avocado_piece_3_pit)` | A piece has each part it has.                 |
| `knowrob:frameName(world:avocado_piece_3, 'avocado_piece_3')` | The tf frame the body is published under, which carries the pose. |

`sdt` and `world` are the prefixes of the annotations' classes and of the annotations
themselves (`Namespace` in the bridge's module). KnowRob runs inside the bridge's
process with its in-memory storage, so it knows the pieces only while the bridge runs.

### Installing KnowRob

KnowRob 2 has no ROS 2 interface, so the bridge uses its Python module. It is built
from source; on Ubuntu 24.04 with ROS 2 Jazzy, four system packages are missing:

```bash
sudo apt install -y libmongoc-dev libraptor2-dev librdf0-dev swi-prolog-nox
git clone --branch dev https://github.com/knowrob/knowrob.git ~/knowrob
cd ~/knowrob && mkdir build && cd build
cmake .. -DCATKIN=OFF -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_MODULE_LIBDIR=dist-packages -DCMAKE_INSTALL_PREFIX=$HOME/.local/knowrob
make -j10 knowrob knowrob_py knowrob-terminal && cmake --install .
echo "$HOME/.local/knowrob/lib/python/dist-packages" \
    > "$(python -c 'import site; print(site.getsitepackages()[0])')/knowrob_local.pth"
```

The last line makes `import knowrob` work in the active Python environment; keep
`~/knowrob`, as KnowRob reads its ontologies and Prolog files from there. The KnowRob
tests in `test_avocado_cutting.py` and in
`test/semantic_digital_twin_test/test_adapters/test_knowrob_bridge.py` are skipped where
KnowRob is not installed.

## Opening the pieces in Isaac Sim

`export_to_usd.py` runs a demonstration headless and writes the pieces it ends with to a
USD file. It needs `usd-core` (`pip install usd-core`, an optional dependency of
`semantic_digital_twin`):

```bash
python export_to_usd.py slicing --seed 9                 # writes avocado_slicing.usda
python export_to_usd.py peeling --output peel.usd         # binary USD instead of text
```

Open the file in Isaac Sim (*File → Open*) and press *Play*. The stage is in metres with
z up and has a physics scene with gravity and a static floor at the height of the
cutting board. Under `/World`, each piece is a rigid body with its mass, centre of mass
and inertia; its parts are meshes that are only seen, and its collision shapes are
convex-hull meshes with the purpose *guide*, so they collide but are not drawn. To write
the pieces at any other moment, from your own code:

```python
from pathlib import Path
from fruit_stage import FruitStage

FruitStage(demonstration.fruit_world, floor_height=0.02).write(Path("pieces.usda"))
```

Only the pieces go into the file: the knife, the poles and the spoon do not, so pieces
the poles held fall onto the floor once the simulation plays, and the segments of a
peeled strip are no longer joined by hinges.

## How it works

MuJoCo cannot split a body while it simulates. So `cutting.py` (`CuttableFruit`) keeps
each part as an exact solid (`manifold3d`). When the knife's edge has gone past the
whole section of a piece in the blade's plane, it cuts those solids along the plane,
removes the piece's body from the model and adds one body per new piece. Then it
recompiles the scene with `MjSpec.recompile`. Each new body carries on from the pose and
velocity of the body it was cut from and collides through convex shapes that stay out
of the other pieces.

- **Knife strokes** (`knife_strokes.py`): straight down, round the pit (as a cook turns
  the fruit against the knife), and scoring the skin to a set depth. The pit stops the
  blade, and flesh within 1.5 mm of the pit need not be cut, as it tears away from a
  real pit.
- **Scooping:** the space the spoon's bowl holds is added up along its real path, every
  2 mm. The flesh in it comes out once the bowl, having been at least 90 % full, comes
  out of the flesh, so a different spoon path gives a different scoop.
- **Peeling:** a strip is cut into segments joined by hinge constraints. While it is
  peeled off, the segments are turned about the hinge lines so that the strip lays
  itself flat.
- **Holding:** a piece reaching into a pole has no joint and stays fixed; a gripped
  piece follows a mocap body.

| File                             | Contents                                                 |
|----------------------------------|----------------------------------------------------------|
| `cutting.py`                     | The cuttable fruit: pieces, parts, cuts, scoops, hinges. |
| `knife_strokes.py`               | The paths the knife's blade takes.                       |
| `random_pit.py`                  | The random pit and the cavity fitted to it.              |
| `fruit_world.py`                 | The pieces as a `semantic_digital_twin` world.           |
| `fruit_stage.py`                 | The pieces of that world as a USD stage for Isaac Sim.   |
| `export_to_usd.py`               | Run a demonstration and write its pieces to USD.         |
| `fruit_world_publisher.py`       | The pieces' world published on ROS 2.                    |
| `publish_on_ros.py`              | Run a demonstration and publish its pieces on ROS 2.     |
| `demonstrations.py`              | The four demonstrations, chosen by name.                 |
| `ask_knowrob.py`                 | Join the published world and ask KnowRob about it.       |
| `solid_avocado_demonstration.py` | What every demo shares: scene, viewer, laying aside.     |
| `held_avocado_demonstration.py`  | The avocado on poles, and halving it.                    |
| `*_demo.py`                      | The four demonstrations.                                 |
| `exceptions.py`                  | The errors the demos raise.                              |

## The avocado model

The model lives in `coraplex/resources/objects/solid_avocado`: `model.xml` (MJCF) with
three watertight meshes in `assets/`, `skin.obj`, `flesh.obj` and `pit.obj`, about
11 cm long. The frame has its origin at the base and its z axis along the long axis to
the stem. Densities are 1050 (skin), 960 (flesh) and 1150 kg/m³ (pit). The model is
BSD-3-Clause licensed; see `NOTICE.md` and `LICENSE.txt` there.

## Tests

`test/coraplex_test/test_avocado_cutting.py` covers cutting, holding, separating,
scooping, hinges, gripping, the random pit, the world, its annotations, storing it in a
database, publishing it on ROS 2, bridging it to KnowRob and the USD stage; the USD
and KnowRob tests are skipped where `usd-core` or KnowRob is not installed. All four demos also run headless in the CI matrix of
`.github/workflows/examples_and_demos.yml`.

```bash
python -m pytest test/coraplex_test/test_avocado_cutting.py
```

## Limits

- Finishing a cut takes a moment (rebuilding the meshes and the scene), so the viewer
  pauses briefly at each cut.
- The meshes of the world's bodies are written to temporary files that are deleted when
  the process exits.
- The knife, the board, the poles and the spoon are not in the `semantic_digital_twin`
  world; only the avocado's pieces are.
