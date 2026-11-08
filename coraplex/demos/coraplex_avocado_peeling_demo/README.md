# Avocado peeling demos

Two demonstrations in MuJoCo, in the kitchen of the IAI apartment
(`semantic_digital_twin/resources/mjcf/iai_apartment.xml`), on a wooden cutting board
on the kitchen island. Both use the peelable avocado of `private_meshes`; its seed 0
comes with CRAM in `coraplex/resources/objects/avocado`, so nothing else is needed.
Both check every stage; a failed check raises one of the exceptions in
`exceptions.py`. With a viewer, the window closes on its own 3 s after the last stage.

```bash
python demo.py            # mocap hands, with viewer
python pr2_demo.py        # the PR2, with viewer
python test_demo.py       # headless, as in CI
python test_pr2_demo.py
python pr2_demo.py /path/to/another/avocado   # or set AVOCADO_DIRECTORY
```

Other avocados come from `scripts/generate_avocado.py` of `private_meshes`.

## The PR2 puts the avocado on the board and peels it (`pr2_demo.py`)

The PR2 stands at the kitchen island, the avocado lying on the island to its right
of the board. It takes the avocado from above with its right hand, lays it on the
board with the stem to the robot's right and moves its hand aside. Its left hand then
holds the fruit down at the base end, on the left, its closed fingertips on top of the
skin, so nothing of it reaches round to the side being peeled. The right hand comes in
across the fruit onto the tip, at the stem, of the upper half's strip facing the robot
and pinches it between its fingertips. It draws the strip a little along the fruit and
then mostly out from it, toward the robot and away from the holding hand, and the
strip tears off tile by tile behind the fingers while the rest of the skin stays on.
Still holding the strip, the right hand backs off, turns to point down, carries the
strip to the island beside the board and lays it down there. Each hand works on the
end on its own side. The run checks that the avocado comes to rest within 5 cm of the
board's centre, that at least 60 % of the strip comes off, that the whole strip lies
within its own length, plus 3 cm, of where it was let go, that no other skin comes off
and that the two arms never touch.

The PR2 is the one of the semantic digital twin. `pr2.py` exports it to MuJoCo when
the run starts, with the meshes, joints and position servos the CRAM simulations
give it, so the ROS 2 workspace holding `iai_pr2_description` must be sourced. Its
base stands fixed, its arms are gravity-compensated, as the real PR2's
counterbalanced arms are, and they move by damped least-squares inverse kinematics
on their servos, each elbow kept on its own side. The hands move in straight lines.
The waiting hand stays to its own side of the board. A hand takes hold with a weld
switched on where it is; the fingers close around what it holds with 1 N at most.
The tips of all strips meet at the stem, so while pinching, the right hand's fingers
touch only the pinched strip and the other arm, which they must not meet. A headless run takes 2.5 to 5 minutes, depending on
the machine's load, for about 40 s of simulated time; the viewer plays at about a
quarter of real time or slower.

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

Both demos are in the CI matrix of `examples_and_demos.yml`.
