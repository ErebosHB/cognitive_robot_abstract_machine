# Avocado peeling demos

Two demonstrations in MuJoCo, in the kitchen of the IAI apartment
(`semantic_digital_twin/resources/mjcf/iai_apartment.xml`), on a wooden cutting board
on the kitchen island. Both use the peelable avocado of `private_meshes`
(`objects/produce/avocado`) and check every stage; a failed check raises one of the
exceptions in `exceptions.py`.

```bash
python demo.py /path/to/private_meshes/objects/produce/avocado        # mocap hands, with viewer
python pr2_demo.py /path/to/private_meshes/objects/produce/avocado    # the PR2, with viewer
AVOCADO_DIRECTORY=/path/to/private_meshes/objects/produce/avocado python test_demo.py
AVOCADO_DIRECTORY=/path/to/private_meshes/objects/produce/avocado python test_pr2_demo.py
```

## The PR2 puts the avocado on the board and peels it (`pr2_demo.py`)

The PR2 stands at the kitchen island, the avocado lying on the island to its right
of the board. It takes the avocado from above with its right hand, carries it over
the board and lays it down. Then its left hand holds the fruit at the base end, and
the right hand pinches the skin at the stem end of the strip facing the robot and
draws it back along the fruit, out and away from the skin. The strip tears off tile
by tile and the rest of the skin stays on. The run checks that the avocado comes to
rest within 5 cm of the board's centre, that at least 60 % of the strip comes off
(its base end stays under the holding hand) and that no other skin comes off.

The PR2 is the one of the semantic digital twin. `pr2.py` exports it to MuJoCo when
the run starts, with the meshes, joints and position servos the CRAM simulations
give it, so the ROS 2 workspace holding `iai_pr2_description` must be sourced. Its
base stands fixed, its arms are gravity-compensated, as the real PR2's
counterbalanced arms are, and they move by damped least-squares inverse kinematics
on their servos. The hands move in straight lines, pointing down. A hand takes hold
with a weld switched on where it is; the fingers close around what it holds with
1 N at most. While it pinches the skin the right hand passes through the fruit, so
it cannot catch the tips of the other strips, which all meet at the stem. A
headless run takes about 2 minutes for 25 s of simulated time; the viewer plays at
about a fifth of real time.

## Mocap hands peel, cut, pit and slice the avocado (`demo.py`)

The avocado lies on the cutting board, the half holding the pit down, and is held
by its upper half. A gripper grabs the skin at the stem end of the upper half's
strip facing up and draws it back over the fruit, tearing that strip off. The
avocado is then cut, and its halves are parted along the cut. The upper half is laid
on its cut face beside the lower one, and the pit is lifted out onto the board. Then
the lower half is cut into its slices and its stem slice is carried away. The
gripper and the holder are mocap bodies. The run checks that at least three
quarters of the strip and no other skin come off, that the pit leaves its cavity by
2 cm, and that the stem slice comes away without its neighbour. A headless run
takes about 40 s.

## The avocado

Its skin is 16 strips of 20 tiles, each tile held to the flesh by an equality
constraint. `peeling.py` releases a tile once it stays lifted 1.5 mm off its seat or
dragged 3 mm along it for 5 steps. The tiles themselves are hidden; each strip is
drawn as one MuJoCo skin per flesh slice that bends with its tiles. The flesh is two
halves of 4 welded slices each; `PeelableFruit.slice` cuts flesh and skin between
slices. The pit collides with the flesh all along. The fruit is attached without a
name prefix, because MuJoCo 3.x does not prefix the bodies a skin follows.

Neither demo is in the CI matrix of `examples_and_demos.yml`, because CI has no
`private_meshes` checkout, and the PR2 demo also needs `iai_pr2_description`.
