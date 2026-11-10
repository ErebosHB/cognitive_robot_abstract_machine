# Avocado peeling demos

Three demonstrations in MuJoCo, in the kitchen of the IAI apartment
(`semantic_digital_twin/resources/mjcf/iai_apartment.xml`), on a wooden cutting board
on the kitchen island. All use the peelable avocado of `private_meshes`; its seed 0
comes with CRAM in `coraplex/resources/objects/avocado`, so nothing else is needed.
All check every stage; a failed check raises one of the exceptions in
`exceptions.py`. With a viewer, the window closes on its own 3 s after the last stage.

```bash
python demo.py                      # mocap hands, with viewer
python pr2_demo.py                  # the PR2 peels one strip, with viewer
python pr2_whole_avocado_demo.py    # the PR2 peels the whole avocado, with viewer
python test_demo.py                 # headless, as in CI
python test_pr2_demo.py
python test_pr2_whole_avocado_demo.py
python pr2_demo.py /path/to/another/avocado   # or set AVOCADO_DIRECTORY
```

Other avocados come from `scripts/generate_avocado.py` of `private_meshes`.

## The PR2 peels the whole avocado (`pr2_whole_avocado_demo.py`)

The PR2 stands at the kitchen island, the avocado lying on the island to its right
of the board. It takes the avocado from above with its right hand, lays it on the
board with the stem to the robot's right and moves its hand aside. Its left hand
takes the fruit by its base end, the fingers pointing along the fruit, and holds it
out above the board. The right hand pinches the skin facing up, leaning a little toward
the robot so it never reaches round the far side of the fruit, ten rows from the held
end, three tiles wide, coming in at a slant from the free end so it stays clear of the
holding fingers. It peels the strip back along the fruit toward the free end and lays
it on the board beside the fruit. The left wrist then turns the fruit about its axis
until skin faces up again, by as little as brings the next column with skin on up
(back a little, when the fruit's sag turned it too far), and
the right hand peels the column with the most skin left of those facing up. Columns
with fewer than 3 tiles left, or where a pinch took fewer than 5, are left. After a full turn
the left hand lowers the fruit until it is just above the board and lets go; the right
hand takes it from above, turns it end for end and lays it down again, and the left
hand picks it up by its peeled stem end. The right hand peels the base end the same
way. The run checks that the avocado comes to rest within 5 cm of the board's centre,
that no skin comes off while it is carried or first picked up, that at most 20 loose
tiles come off while it is turned end for end, and again while it is picked up by its
other end, that at least 90 % of the skin comes off in
all and that the two arms never touch.

## The PR2 peels one strip (`pr2_demo.py`)

The same as the whole avocado, up to the first strip: the PR2 lays the avocado on the
board, picks it up by its base end and peels one strip, one tile wide, which it lays
on the board. The strip starts ten rows from the base, below the end rows, whose tiles
each run into two columns, where a strip would fork. The run checks that at least 15
tiles still hang together on the fingers when the strip is laid down.

The PR2 is the one of the semantic digital twin. `pr2.py` exports it to MuJoCo when
the run starts, with the meshes, joints and position servos the CRAM simulations
give it, so the ROS 2 workspace holding `iai_pr2_description` must be sourced. Its
base stands fixed, its arms are gravity-compensated, as the real PR2's
counterbalanced arms are, and they move by damped least-squares inverse kinematics
on their servos, each elbow kept on its own side. The hands move in straight lines.
The waiting hand stays to its own side of the board. A hand takes hold with a weld
switched on where it is; the fingers close around what it holds with 1 N at most.
The fingertips close along the strip, so the hand is narrow across it; while pinching,
they touch only the pinched tiles and the other arm, which they must not meet. The
left hand holds the fruit by welds to its twelve heaviest flesh chunks, and the skin's
links are three times as strong as the model's default, 30 N/m, so strips come off
long. A headless run of `pr2_demo.py` takes about 5 minutes; one of
`pr2_whole_avocado_demo.py` about 37 minutes. The viewer plays at roughly a tenth of
real time.

## Mocap hands peel, cut, pit and slice the avocado (`demo.py`)

The avocado lies on the cutting board, the half holding the pit down, and is held
by its upper half. A gripper grabs a tile near the stem of the upper half's column
facing up and draws it back over the fruit, so a strip peels off from there toward
the base. The avocado is then halved, and its halves are parted along the cut. The
upper half is laid on its cut face beside the lower one, and the pit is lifted out
onto the board. Then the fruit is sliced across near the stem and the lower half's
stem-side piece is carried away. The gripper and the holder are 50 g hands that a
soft weld drags after mocap targets, and that grip with a stiff weld. The run checks
that at least 15 tiles come off as one strip and no other skin comes off, that the pit
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

All three demos are in the CI matrix of `examples_and_demos.yml`.
