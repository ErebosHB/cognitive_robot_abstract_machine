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
across the fruit onto a tile near the stem of the upper half's column facing the robot
and pinches it between its fingertips. It draws the strip a little along the fruit and
then mostly out from it, toward the robot and away from the holding hand, and the
strip tears off tile by tile behind the fingers while the rest of the skin stays on.
Still holding the strip, the right hand backs off, turns to point down, carries the
strip to the island beside the board and lays it down there. Each hand works on the
end on its own side. The run checks that the avocado comes to rest within 5 cm of the
board's centre, that at least 8 tiles come off as one strip, that what still hangs on
the pinched tile lies within the column's length, plus 3 cm, of where it was let go, that no other skin comes off
and that the two arms never touch.

The PR2 is the one of the semantic digital twin. `pr2.py` exports it to MuJoCo when
the run starts, with the meshes, joints and position servos the CRAM simulations
give it, so the ROS 2 workspace holding `iai_pr2_description` must be sourced. Its
base stands fixed, its arms are gravity-compensated, as the real PR2's
counterbalanced arms are, and they move by damped least-squares inverse kinematics
on their servos, each elbow kept on its own side. The hands move in straight lines.
The waiting hand stays to its own side of the board. A hand takes hold with a weld
switched on where it is; the fingers close around what it holds with 1 N at most.
Many tiles meet near the stem, so while pinching, the right hand's fingers touch only
the pinched tile and the other arm, which they must not meet. A headless run takes
about 7 minutes for about 40 s of simulated time; the viewer plays at roughly a tenth
of real time.

## Mocap hands peel, cut, pit and slice the avocado (`demo.py`)

The avocado lies on the cutting board, the half holding the pit down, and is held
by its upper half. A gripper grabs a tile near the stem of the upper half's column
facing up and draws it back over the fruit, so a strip peels off from there toward
the base. The avocado is then halved, and its halves are parted along the cut. The
upper half is laid on its cut face beside the lower one, and the pit is lifted out
onto the board. Then the fruit is sliced across near the stem and the lower half's
stem-side piece is carried away. The gripper and the holder are 50 g hands that a
soft weld drags after mocap targets, and that grip with a stiff weld. The run checks
that at least 8 tiles come off as one strip and no other skin comes off, that the pit
leaves its cavity by 2 cm, and that the stem-side piece comes away without the rest
of the half. A headless run takes about 75 s.

## The avocado

The avocado is made of free bodies held together by equality constraints: 168 flesh
chunks (14 slabs across the fruit, 12 wedges around it) welded to their neighbours,
708 skin tiles each welded to the chunk under it, and the pit welded into its seat.
`peeling.py` switches these constraints, since MuJoCo has none that break. A tile
comes off when its weld stays loaded past the skin's hold on the flesh, about 0.2 N
pulling outward at an edge. Neighbouring tiles are linked by hinges that switch on
once one of them is off and tear past their strength; links along a column, stem to
base, are stronger than links across, so a peel runs as a strip. `PeelableFruit.cut`,
`halve` and `slice` cut the flesh and skin along a knife plane, landing on the
nearest boundary between chunks, and every separate piece then collides with the
others. The pit collides with the flesh all along. Grab tiles with a dynamic body,
such as a robot finger or a dragged hand, never a mocap body welded straight to a
0.05 g tile. The scene uses elliptic friction cones, which the fruit asks for. With
about 5300 degrees of freedom the fruit alone simulates at about a fifth of real
time.

Both demos are in the CI matrix of `examples_and_demos.yml`.
