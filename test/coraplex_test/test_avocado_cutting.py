"""
Coverage for cutting the solid avocado of the avocado demos along a knife's plane.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import manifold3d
import mujoco
import numpy as np
import pytest
from typing_extensions import Any, Iterator

# %% the demo's modules and the scene


@dataclass
class DemoPaths:
    """
    Where the avocado demos and the solid avocado lie in this repository.
    """

    repository: Path = Path(__file__).resolve().parents[2]
    """
    The root of the repository.
    """

    def demo_directory(self) -> Path:
        """
        :return: The folder of the avocado demos, whose modules import each other by
            their bare names.
        """
        return self.repository / "coraplex" / "demos" / "coraplex_solid_avocado_demo"

    def solid_avocado(self) -> Path:
        """
        :return: The MuJoCo model of the solid avocado.
        """
        return (
            self.repository
            / "coraplex"
            / "resources"
            / "objects"
            / "solid_avocado"
            / "model.xml"
        )


@dataclass
class KnifeInScene:
    """
    The knife the tests put next to the fruit: a mocap body holding the blade's site.
    """

    site_name: str = "blade_edge"
    """
    The name of the site in the middle of the cutting edge.
    """

    edge_length: float = 0.2
    """
    The length of the cutting edge in metres.
    """

    def add_to(self, spec: mujoco.MjSpec) -> None:
        """
        :param spec: The scene to add the knife to, far above the fruit.
        """
        knife = spec.worldbody.add_body(name="knife", mocap=True, pos=[0, 0, 1])
        knife.add_site(name=self.site_name)


@dataclass
class SpoonInScene:
    """
    The spoon the scooping tests lead through the fruit: a mocap body whose frame is
    in the middle of a small bowl.
    """

    body_name: str = "spoon"
    """
    The name of the spoon's body.
    """

    bowl_size: tuple[float, float, float] = (0.006, 0.004, 0.0015)
    """
    Half the length, width and thickness of the bowl, in metres.
    """

    reach: float = 0.004
    """
    How far beyond its back, on its hollow side, the bowl holds what it scoops, in
    metres.
    """

    def add_to(self, spec: mujoco.MjSpec) -> None:
        """
        :param spec: The scene to add the spoon to, far above the fruit.
        """
        spec.worldbody.add_body(name=self.body_name, mocap=True, pos=[0, 0, 1])


@pytest.fixture(scope="module")
def cutting() -> Iterator[ModuleType]:
    """
    The ``cutting`` module of the avocado demos, importable only from their folder.
    """
    demo_modules = (
        "cutting",
        "exceptions",
        "knife_strokes",
        "solid_avocado_demonstration",
        "held_avocado_demonstration",
        "scooping_demo",
        "random_pit",
        "fruit_world",
        "fruit_stage",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(DemoPaths().demo_directory()))
        for name in demo_modules:
            patch.delitem(sys.modules, name, raising=False)
        yield importlib.import_module("cutting")
        for name in demo_modules:
            sys.modules.pop(name, None)


@pytest.fixture
def knife_strokes(cutting: ModuleType) -> ModuleType:
    """
    The ``knife_strokes`` module of the avocado demos, imported while :func:`cutting`
    makes their folder importable.
    """
    return importlib.import_module("knife_strokes")


@pytest.fixture
def scooping_demo(cutting: ModuleType) -> ModuleType:
    """
    The ``scooping_demo`` module of the avocado demos, imported while :func:`cutting`
    makes their folder importable.
    """
    return importlib.import_module("scooping_demo")


@pytest.fixture
def knife() -> KnifeInScene:
    """
    The knife of the scene.
    """
    return KnifeInScene()


def floating_fruit(
    cutting: ModuleType,
    knife: KnifeInScene,
    holders: list[manifold3d.Manifold],
    spoon: SpoonInScene | None = None,
) -> Any:
    """
    :param cutting: The ``cutting`` module.
    :param knife: The knife to put next to the fruit.
    :param holders: The solids that hold the fruit in place.
    :param spoon: The spoon to put next to the fruit too, if any.
    :return: The whole solid avocado floating without gravity, its frame at the world's
        origin with the long axis along z, next to the knife.
    """
    spec = mujoco.MjSpec.from_file(str(DemoPaths().solid_avocado()))
    spec.option.gravity = [0, 0, 0]
    knife.add_to(spec)
    if spoon is not None:
        spoon.add_to(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return cutting.CuttableFruit(spec, model, data, "avocado", holders=holders)


@pytest.fixture
def fruit(cutting: ModuleType, knife: KnifeInScene) -> Any:
    """
    The whole solid avocado of :func:`floating_fruit`, free.
    """
    return floating_fruit(cutting, knife, [])


@pytest.fixture
def pole() -> manifold3d.Manifold:
    """
    A pole stuck sideways into the avocado near its base, from the side the y axis
    points away from, through skin and flesh and clear of the pit.
    """
    return (
        manifold3d.Manifold.cylinder(0.05, 0.003, 0.003, 24)
        .rotate([90.0, 0.0, 0.0])
        .translate([0.0, -0.012, 0.02])
    )


@pytest.fixture
def held_fruit(
    cutting: ModuleType, knife: KnifeInScene, pole: manifold3d.Manifold
) -> Any:
    """
    The whole solid avocado of :func:`floating_fruit`, held by :func:`pole`.
    """
    return floating_fruit(cutting, knife, [pole])


@pytest.fixture
def slanted_plane(cutting: ModuleType) -> Any:
    """
    A plane through the avocado and its pit, square to none of its axes.
    """
    return cutting.CuttingPlane([0.004, 0.001, 0.05], [1.0, 0.3, 0.4])


# %% the whole fruit


def test_fruit_is_one_body_of_three_materials_until_cut(cutting, fruit):
    """
    The uncut fruit is a single body holding a part of every material.
    """
    assert fruit.model.nbody == 3
    assert [piece.body_name for piece in fruit.pieces] == [fruit.body_name]
    assert fruit.pieces[0].materials() == set(cutting.FruitMaterial)


def test_piece_weighs_what_mujoco_weighs_its_meshes(fruit):
    """
    A piece's mass and centre of mass match what MuJoCo computes from the meshes.
    """
    piece = fruit.pieces[0]
    body_id = fruit.body_id(piece)
    distribution = piece.mass_distribution()
    assert distribution.mass == pytest.approx(fruit.model.body_mass[body_id], rel=1e-5)
    assert distribution.centre == pytest.approx(
        fruit.model.body_ipos[body_id], abs=1e-6
    )


# %% cutting along a plane


def test_cut_surfaces_lie_in_the_plane(fruit, slanted_plane):
    """
    Every cut part ends exactly on the cutting plane, one piece on each side.
    """
    made = fruit.cut(slanted_plane)
    assert len(made) == 2
    sides = set()
    for piece in made:
        for part in piece.parts:
            if part.material in fruit.uncuttable:
                continue
            vertices = part.triangle_mesh_of(part.solid).vertices
            heights = (vertices - slanted_plane.point) @ slanted_plane.normal
            nearest, farthest = sorted(np.abs([heights.min(), heights.max()]))
            assert nearest == pytest.approx(0.0, abs=1e-6)
            assert farthest > 0.01
            sides.add(bool(heights.mean() > 0))
    assert sides == {True, False}


def test_cut_replaces_the_body_by_one_for_each_piece(fruit, slanted_plane):
    """
    A cut swaps the fruit's body for one body per piece and keeps the mass.
    """
    whole_mass = fruit.model.body_mass[fruit.body_id(fruit.pieces[0])]
    made = fruit.cut(slanted_plane)
    assert made == fruit.pieces
    assert fruit.model.nbody == 4
    assert fruit.body_id(made[0]) != fruit.body_id(made[1])
    assert (
        mujoco.mj_name2id(fruit.model, mujoco.mjtObj.mjOBJ_BODY, fruit.body_name) == -1
    )
    piece_masses = [fruit.model.body_mass[fruit.body_id(piece)] for piece in made]
    assert sum(piece_masses) == pytest.approx(whole_mass, rel=1e-4)


def test_pit_is_not_cut(cutting, fruit, slanted_plane):
    """
    A plane through the pit leaves the pit whole in one piece.
    """
    pit = cutting.FruitMaterial.PIT
    pit_volume = fruit.pieces[0].part_of(pit).solid.volume()
    made = fruit.cut(slanted_plane)
    with_pit = [piece for piece in made if pit in piece.materials()]
    assert with_pit == [fruit.piece_with(pit)]
    assert with_pit[0].part_of(pit).solid.volume() == pytest.approx(pit_volume)


def test_plane_beside_the_fruit_cuts_nothing(cutting, fruit):
    """
    A plane that misses the fruit leaves the model untouched.
    """
    model = fruit.model
    assert fruit.cut(cutting.CuttingPlane([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])) == []
    assert fruit.model is model
    assert len(fruit.pieces) == 1


def test_nothing_smaller_than_the_smallest_piece_is_cut_off(cutting, fruit):
    """
    A plane grazing the stem end, cutting off a sliver, does not cut.
    """
    highest = fruit.pieces[0].outline.bounds[1][2]
    grazing = cutting.CuttingPlane([0.0, 0.0, highest - 0.001], [0.0, 0.0, 1.0])
    assert fruit.cut(grazing) == []


def test_pieces_carry_on_from_the_pose_and_velocity_of_the_whole(fruit, slanted_plane):
    """
    The pieces start where the cut body was, moving as it moved.
    """
    velocity = np.array([0.1, -0.2, 0.3, 0.5, -0.4, 0.2])
    fruit.data.qpos[:3] = [0.2, -0.1, 0.4]
    fruit.data.qvel[:6] = velocity
    mujoco.mj_forward(fruit.model, fruit.data)
    before = fruit.pose_of(fruit.pieces[0])
    for piece in fruit.cut(slanted_plane):
        after = fruit.pose_of(piece)
        assert after.position == pytest.approx(before.position)
        assert after.quaternion == pytest.approx(before.quaternion)
        assert after.velocity == pytest.approx(velocity)


def test_cut_pieces_do_not_push_each_other_apart(fruit, slanted_plane):
    """
    Without gravity the pieces of a cut through the pit stay where they are.
    """
    made = fruit.cut(slanted_plane)
    body_ids = [fruit.body_id(piece) for piece in made]
    start = fruit.data.xipos[body_ids].copy()
    for _ in range(250):
        mujoco.mj_step(fruit.model, fruit.data)
    moved = np.linalg.norm(fruit.data.xipos[body_ids] - start, axis=1)
    assert moved.max() < 0.002


def test_second_cut_cuts_every_piece_it_crosses(cutting, fruit, slanted_plane):
    """
    A second plane cuts both pieces of the first cut.
    """
    fruit.cut(slanted_plane)
    made = fruit.cut(cutting.CuttingPlane([0.0, 0.0, 0.06], [0.0, 0.0, 1.0]))
    assert len(made) == 4
    assert made == fruit.pieces


# %% cutting with a blade


def test_blade_part_way_in_leaves_the_fruit_whole(cutting, fruit, knife):
    """
    A blade whose edge is still inside the fruit has not cut it.
    """
    blade = cutting.KnifeBlade(knife.site_name, knife.edge_length)
    # The blade lies in the plane z = 0.05 and moves along -y; its edge is at the axis.
    fruit.data.mocap_pos[0] = [0.0, 0.0, 0.05]
    mujoco.mj_forward(fruit.model, fruit.data)
    assert fruit.cut_with(blade) == []
    assert len(fruit.pieces) == 1


def test_blade_gone_through_cuts_along_its_plane(cutting, fruit, knife):
    """
    A blade whose edge has left the fruit cuts it along the blade's plane.
    """
    blade = cutting.KnifeBlade(knife.site_name, knife.edge_length)
    fruit.data.mocap_pos[0] = [0.0, -0.06, 0.05]
    mujoco.mj_forward(fruit.model, fruit.data)
    made = fruit.cut_with(blade)
    assert len(made) == 2
    skin = cutting.FruitMaterial.SKIN
    skin_bounds = [
        piece.part_of(skin).triangle_mesh_of(piece.part_of(skin).solid).bounds
        for piece in made
    ]
    lower, upper = sorted(skin_bounds, key=lambda bounds: bounds[0][2])
    assert lower[1][2] == pytest.approx(0.05, abs=1e-6)
    assert upper[0][2] == pytest.approx(0.05, abs=1e-6)


def test_blade_shorter_than_the_cut_leaves_the_fruit_whole(cutting, fruit, knife):
    """
    An edge shorter than the fruit is wide does not cut the fruit through.
    """
    blade = cutting.KnifeBlade(knife.site_name, 0.02)
    fruit.data.mocap_pos[0] = [0.0, -0.06, 0.05]
    mujoco.mj_forward(fruit.model, fruit.data)
    assert fruit.cut_with(blade) == []


def test_cut_adds_up_over_the_blade_poses_of_one_plane(cutting, fruit, knife):
    """
    A blade that goes past one half of the section and then, turned round in the same
    plane, past the other half cuts the fruit, as a knife going round the pit does.
    """
    blade = cutting.KnifeBlade(knife.site_name, knife.edge_length)
    # The edge lies on the fruit's axis, first cutting towards -y, then towards +y.
    fruit.data.mocap_pos[0] = [0.0, 0.0, 0.05]
    mujoco.mj_forward(fruit.model, fruit.data)
    assert fruit.cut_with(blade) == []
    fruit.data.mocap_quat[0] = [0.0, 0.0, 0.0, 1.0]
    mujoco.mj_forward(fruit.model, fruit.data)
    assert len(fruit.cut_with(blade)) == 2


def test_blade_in_another_plane_starts_the_cut_over(cutting, fruit, knife):
    """
    What a blade went past in one plane does not count for a cut in another.
    """
    blade = cutting.KnifeBlade(knife.site_name, knife.edge_length)
    fruit.data.mocap_pos[0] = [0.0, 0.0, 0.05]
    mujoco.mj_forward(fruit.model, fruit.data)
    fruit.cut_with(blade)
    fruit.data.mocap_pos[0] = [0.0, 0.0, 0.06]
    fruit.data.mocap_quat[0] = [0.0, 0.0, 0.0, 1.0]
    mujoco.mj_forward(fruit.model, fruit.data)
    assert fruit.cut_with(blade) == []
    assert len(fruit.pieces) == 1


# %% the pit in the blade's way


def test_pit_outline_lies_on_the_pit_in_the_plane(cutting, fruit):
    """
    The outline a plane cuts out of the pit lies in the plane, on the pit's surface.
    """
    plane = cutting.CuttingPlane([0.0, 0.0, 0.05], [0.0, 0.0, 1.0])
    outline = fruit.section_outline(plane, fruit.uncuttable)
    pit = fruit.pieces[0].part_of(cutting.FruitMaterial.PIT)
    pit_mesh = pit.triangle_mesh_of(pit.solid)
    assert len(outline) > 0
    assert (outline - plane.point) @ plane.normal == pytest.approx(0.0, abs=1e-6)
    _, distances, _ = pit_mesh.nearest.on_surface(outline)
    # Slicing simplifies the outline by a few hundredths of a millimetre.
    assert distances.max() < 5e-5


def test_room_ahead_of_the_blade_ends_at_the_pit(cutting, fruit, knife):
    """
    A blade cutting through the pit's middle can move on until its edge reaches the
    pit's outline in its plane.
    """
    blade = cutting.KnifeBlade(knife.site_name, knife.edge_length)
    fruit.data.mocap_pos[0] = [0.0, 0.06, 0.05]
    mujoco.mj_forward(fruit.model, fruit.data)
    outline = fruit.section_outline(
        blade.plane(fruit.model, fruit.data), fruit.uncuttable
    )
    room = blade.room_ahead(fruit.model, fruit.data, outline)
    assert room == pytest.approx(0.06 - outline[:, 1].max())


def test_blade_above_the_pit_has_room_all_the_way(cutting, fruit, knife):
    """
    A blade in a plane that misses the pit has nothing in its way.
    """
    blade = cutting.KnifeBlade(knife.site_name, knife.edge_length)
    highest = fruit.pieces[0].outline.bounds[1][2]
    fruit.data.mocap_pos[0] = [0.0, 0.06, highest - 0.01]
    mujoco.mj_forward(fruit.model, fruit.data)
    outline = fruit.section_outline(
        blade.plane(fruit.model, fruit.data), fruit.uncuttable
    )
    assert blade.room_ahead(fruit.model, fruit.data, outline) == float("inf")


def test_section_to_cut_leaves_the_pit_out(cutting, fruit):
    """
    The points a blade has to go past lie in its plane, in the fruit, and none in the
    pit.
    """
    piece = fruit.pieces[0]
    plane = cutting.CuttingPlane([0.0, 0.0, 0.05], [0.0, 0.0, 1.0])
    samples = fruit.incision_into(piece, plane).samples
    pit = piece.part_of(cutting.FruitMaterial.PIT)
    assert (samples - plane.point) @ plane.normal == pytest.approx(0.0, abs=1e-6)
    assert not pit.triangle_mesh_of(pit.solid).contains(samples).any()
    _, distances, _ = piece.outline.nearest.on_surface(samples)
    assert np.all(piece.outline.contains(samples) | (distances < 1e-6))


# %% held pieces


def test_fruit_reaching_into_a_holder_is_held(held_fruit):
    """
    A fruit something is stuck into is fixed in place: its body has no joint.
    """
    assert held_fruit.is_held(held_fruit.pieces[0])
    assert held_fruit.model.njnt == 0


def test_piece_cut_off_clear_of_the_holders_is_free(cutting, held_fruit):
    """
    Of the pieces of a cut through every material, only the one the pole reaches into
    stays held.
    """
    held_fruit.cut(cutting.CuttingPlane([0.0, 0.0, 0.085], [0.0, 0.0, 1.0]))
    lowest_first = sorted(
        held_fruit.pieces,
        key=lambda piece: held_fruit.data.xipos[held_fruit.body_id(piece)][2],
    )
    assert [held_fruit.is_held(piece) for piece in lowest_first] == [True, False]


def test_pieces_of_a_cut_through_the_skin_only_stay_held(cutting, held_fruit):
    """
    Scoring the skin of a held fruit leaves the scored-off skin held too, as it stays
    stuck to the flesh, even where the pole does not reach into it.
    """
    skin = cutting.FruitMaterial.SKIN
    plane = cutting.CuttingPlane([0.0, 0.0, 0.05], [0.0, 1.0, 0.0])
    made = held_fruit.cut(plane, materials=frozenset({skin}))
    assert sorted(len(piece.materials()) for piece in made) == [1, 3]
    assert all(held_fruit.is_held(piece) for piece in made)
    skin_only = next(piece for piece in made if piece.materials() == {skin})
    assert not held_fruit.reaches_holder(skin_only, held_fruit.pose_of(skin_only))


def test_parts_left_uncut_stay_together(cutting, fruit):
    """
    A plane just beside the axis holds most of the flesh on one side and most of the
    pit on the other; scoring the skin along it leaves flesh and pit in one piece.
    """
    materials = cutting.FruitMaterial
    plane = cutting.CuttingPlane([0.002, 0.0, 0.05], [1.0, 0.0, 0.0])
    made = fruit.cut(plane, materials=frozenset({materials.SKIN}))
    assert sorted(piece.materials() for piece in made) == sorted(
        [{materials.SKIN}, set(materials)]
    )


def test_released_piece_moves_freely_from_where_it_was(held_fruit):
    """
    Releasing a held piece gives its body a free joint and leaves it where it was.
    """
    piece = held_fruit.pieces[0]
    position = held_fruit.data.xpos[held_fruit.body_id(piece)].copy()
    held_fruit.release(piece)
    assert not held_fruit.is_held(piece)
    assert held_fruit.data.xpos[held_fruit.body_id(piece)] == pytest.approx(position)


def test_held_piece_stays_where_it_is(fruit):
    """
    Holding a moving piece fixes it where it is.
    """
    piece = fruit.pieces[0]
    fruit.data.qvel[:3] = [0.1, 0.0, 0.0]
    for _ in range(10):
        mujoco.mj_step(fruit.model, fruit.data)
    position = fruit.data.xpos[fruit.body_id(piece)].copy()
    fruit.hold(piece)
    for _ in range(10):
        mujoco.mj_step(fruit.model, fruit.data)
    assert fruit.is_held(piece)
    assert fruit.data.xpos[fruit.body_id(piece)] == pytest.approx(position)


# %% strokes round an outline


def test_circuit_edge_faces_the_outline_at_its_clearance(knife_strokes):
    """
    Going round an outline, the cutting edge faces the outline's middle and keeps the
    clearance from it; a negative clearance takes it inside.
    """
    square = np.array([[x, y, 0.0] for x in (-0.01, 0.01) for y in (-0.01, 0.01)])
    for clearance in (0.002, -0.003):
        circuit = knife_strokes.OutlineCircuit(
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            square,
            clearance,
        )
        direction = circuit.direction(np.pi / 2)
        distance = circuit.edge_distance(direction)
        blade = circuit.blade_pose(direction, distance)
        assert direction == pytest.approx([0.0, 1.0, 0.0])
        assert distance == pytest.approx(0.01 + clearance)
        assert blade.edge == pytest.approx(distance * direction)
        assert blade.into_blade == pytest.approx(direction)


# %% taking a part out


def test_separated_pit_is_a_piece_of_its_own_where_it_was(cutting, fruit):
    """
    Taking the pit out makes it a piece that lies where the pit lay.
    """
    pit = cutting.FruitMaterial.PIT
    whole = fruit.pieces[0]
    pit_centre = whole.part_of(pit).mass_distribution().centre
    separated = fruit.separate(whole, pit)
    assert separated.materials() == {pit}
    assert len(fruit.pieces) == 2
    assert fruit.piece_with(pit) is separated
    assert fruit.data.xipos[fruit.body_id(separated)] == pytest.approx(
        pit_centre, abs=1e-6
    )
    remaining = next(piece for piece in fruit.pieces if piece is not separated)
    assert remaining.materials() == set(cutting.FruitMaterial) - {pit}


def test_missing_part_is_reported(cutting, fruit):
    """
    Asking a piece for a material it lacks raises the fitting exception.
    """
    pit = cutting.FruitMaterial.PIT
    separated = fruit.separate(fruit.pieces[0], pit)
    exceptions = sys.modules["exceptions"]
    with pytest.raises(exceptions.FruitPartNotFound):
        separated.part_of(cutting.FruitMaterial.SKIN)


# %% scooping a region out


@pytest.fixture
def scoop() -> manifold3d.Manifold:
    """
    A small ball in the flesh between the pit and the stem, clear of skin and pit.
    """
    return manifold3d.Manifold.sphere(0.006, 32).translate([0.0, 0.0, 0.085])


def test_carved_flesh_comes_out_as_a_free_piece(cutting, held_fruit, scoop):
    """
    Carving a region out of a held fruit takes the flesh in it out as a free piece and
    leaves the rest held, the volume of the two adding up to the whole.
    """
    flesh = cutting.FruitMaterial.FLESH
    whole = held_fruit.pieces[0]
    whole_volume = whole.volume()
    expected = (whole.part_of(flesh).solid ^ scoop).volume()
    carved = held_fruit.carve(whole, scoop, frozenset({flesh}))
    rest = next(piece for piece in held_fruit.pieces if piece is not carved)
    assert carved.materials() == {flesh}
    assert carved.volume() == pytest.approx(expected)
    assert not held_fruit.is_held(carved)
    assert held_fruit.is_held(rest)
    assert rest.volume() + carved.volume() == pytest.approx(whole_volume)


def test_region_beside_the_fruit_carves_nothing(cutting, fruit):
    """
    A region that holds no fruit leaves the fruit whole.
    """
    beside = manifold3d.Manifold.sphere(0.006, 32).translate([0.2, 0.0, 0.05])
    assert fruit.carve(fruit.pieces[0], beside) is None
    assert len(fruit.pieces) == 1


# %% hinged segments


@pytest.fixture
def hinge_planes(cutting: ModuleType) -> list[Any]:
    """
    Two planes across the avocado's long axis.
    """
    return [
        cutting.CuttingPlane([0.0, 0.0, height], [0.0, 0.0, 1.0])
        for height in (0.04, 0.07)
    ]


@pytest.fixture
def hinged_fruit(fruit: Any, hinge_planes: list[Any]) -> list[Any]:
    """
    The whole avocado hinged along :func:`hinge_planes`, the pit whole in the middle
    segment.
    """
    return fruit.hinge(fruit.pieces[0], hinge_planes)


def test_hinge_cuts_segments_joined_along_each_cut(fruit, hinge_planes, hinged_fruit):
    """
    Hinging the avocado along two planes cuts it into three segments, each joined to
    the next about a line in the plane between them.
    """
    assert len(hinged_fruit) == 3
    for first, second, plane in zip(hinged_fruit, hinged_fruit[1:], hinge_planes):
        for end in fruit.hinge_ends(first, second):
            assert (end - plane.point) @ plane.normal == pytest.approx(0.0, abs=1e-9)


def test_hinged_segments_stay_joined_while_they_turn(fruit, hinged_fruit):
    """
    Segments set spinning keep each hinge's ends together.
    """
    for number, segment in enumerate(hinged_fruit):
        first_dof = fruit.model.body_dofadr[fruit.body_id(segment)]
        fruit.data.qvel[first_dof + 3 : first_dof + 6] = [0.0, 0.5 * number, 0.2]
    for _ in range(100):
        mujoco.mj_step(fruit.model, fruit.data)
    model, data = fruit.model, fruit.data
    for joint in range(model.neq):
        ends = [
            data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ anchor
            for body_id, anchor in (
                (model.eq_obj1id[joint], model.eq_data[joint][:3]),
                (model.eq_obj2id[joint], model.eq_data[joint][3:6]),
            )
        ]
        assert np.linalg.norm(ends[0] - ends[1]) < 1e-3


def test_pieces_not_hinged_have_no_hinge_ends(fruit, hinged_fruit):
    """
    Asking for the hinge between segments that are not neighbours raises the fitting
    exception.
    """
    exceptions = sys.modules["exceptions"]
    with pytest.raises(exceptions.PiecesNotHinged):
        fruit.hinge_ends(hinged_fruit[0], hinged_fruit[2])


# %% gripping a piece


def test_gripped_piece_goes_where_its_mocap_body_is_put(fruit):
    """
    A gripped piece is held, and moves to wherever its mocap body is put.
    """
    piece = fruit.pieces[0]
    fruit.grip(piece)
    mocap_id = fruit.model.body_mocapid[fruit.body_id(piece)]
    fruit.data.mocap_pos[mocap_id] = [0.01, 0.02, 0.03]
    mujoco.mj_step(fruit.model, fruit.data)
    assert fruit.is_held(piece)
    assert fruit.pose_of(piece).position == pytest.approx([0.01, 0.02, 0.03])


def test_released_gripped_piece_moves_freely(fruit):
    """
    Releasing a gripped piece frees it where it is.
    """
    piece = fruit.pieces[0]
    fruit.grip(piece)
    fruit.release(piece)
    assert not fruit.is_held(piece)
    assert fruit.model.body_mocapid[fruit.body_id(piece)] == -1


# %% scooping along a spoon's way


@pytest.fixture
def spoon() -> SpoonInScene:
    """
    The spoon of the scene.
    """
    return SpoonInScene()


@pytest.fixture
def fruit_with_spoon(
    cutting: ModuleType, knife: KnifeInScene, spoon: SpoonInScene
) -> Any:
    """
    The whole solid avocado of :func:`floating_fruit`, free, next to :func:`spoon`.
    """
    return floating_fruit(cutting, knife, [], spoon)


@dataclass
class SpoonWay:
    """
    A straight way for the spoon along the x axis through the flesh between the pit
    and the stem, its bowl's back facing the negative y axis.
    """

    height: float = 0.085
    """
    How far along the avocado's long axis the way lies, in metres.
    """

    side: float = -0.008
    """
    Where along the y axis the way lies, in metres.
    """

    step_length: float = 0.0005
    """
    How far the spoon moves per step, in metres.
    """

    def lead(
        self,
        cutting: ModuleType,
        fruit: Any,
        spoon: SpoonInScene,
        start: float,
        end: float,
    ) -> list[list[Any]]:
        """
        :param cutting: The ``cutting`` module.
        :param fruit: The fruit next to the spoon.
        :param spoon: The spoon.
        :param start: Where along the x axis the spoon starts, in metres.
        :param end: Where it ends.
        :return: The scoops of flesh each step made.
        """
        bowl = cutting.SpoonBowl(spoon.body_name, spoon.bowl_size, spoon.reach)
        mocap_id = fruit.model.body_mocapid[
            mujoco.mj_name2id(fruit.model, mujoco.mjtObj.mjOBJ_BODY, spoon.body_name)
        ]
        # The bowl's back, its z axis, along the world's negative y axis.
        fruit.data.mocap_quat[mocap_id] = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
        made = []
        for along in np.arange(start, end, self.step_length):
            fruit.data.mocap_pos[mocap_id] = [along, self.side, self.height]
            mujoco.mj_forward(fruit.model, fruit.data)
            made.append(
                fruit.scoop_with(
                    bowl, materials=frozenset({cutting.FruitMaterial.FLESH})
                )
            )
        return made


def test_spoon_led_through_the_flesh_scoops_what_its_bowl_held(
    cutting, fruit_with_spoon, spoon
):
    """
    A spoon led through the flesh and out again scoops nothing until it comes out,
    then the flesh its bowl held on the way, and nothing else.
    """
    way = SpoonWay()
    made = way.lead(cutting, fruit_with_spoon, spoon, -0.05, 0.05)
    scooping_steps = [number for number, scoops in enumerate(made) if scoops]
    assert len(scooping_steps) == 1
    (scooped,) = made[scooping_steps[0]]
    assert scooped.materials() == {cutting.FruitMaterial.FLESH}
    assert not fruit_with_spoon.is_held(scooped)
    corners = fruit_with_spoon.pose_of(scooped).to_world(scooped.outline.vertices)
    _, width, thickness = spoon.bowl_size
    tolerance = 1e-4
    assert corners[:, 1].min() >= way.side - thickness - tolerance
    assert corners[:, 1].max() <= way.side + thickness + spoon.reach + tolerance
    assert np.abs(corners[:, 2] - way.height).max() <= width + tolerance


def test_spoon_stopped_in_the_flesh_scoops_nothing(cutting, fruit_with_spoon, spoon):
    """
    A spoon that goes into the flesh but does not come out again scoops nothing.
    """
    made = SpoonWay().lead(cutting, fruit_with_spoon, spoon, -0.05, 0.0)
    assert not any(made)
    assert len(fruit_with_spoon.pieces) == 1


def test_spoon_beside_the_fruit_scoops_nothing(cutting, fruit_with_spoon, spoon):
    """
    A spoon led past the fruit scoops nothing.
    """
    beside = SpoonWay(side=-0.1)
    made = beside.lead(cutting, fruit_with_spoon, spoon, -0.05, 0.05)
    assert not any(made)
    assert len(fruit_with_spoon.pieces) == 1


# %% what stops a spoon


def bowl_at(cutting: ModuleType, spoon: SpoonInScene, position: np.ndarray) -> Any:
    """
    :param cutting: The ``cutting`` module.
    :param spoon: The spoon.
    :param position: Where the middle of the bowl would be, in metres.
    :return: The bowl and its pose there, unturned.
    """
    bowl = cutting.SpoonBowl(spoon.body_name, spoon.bowl_size, spoon.reach)
    return bowl, cutting.BodyPose(position, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(6))


def test_bowl_in_the_flesh_is_not_blocked(cutting, fruit, spoon):
    """
    A bowl wholly in the flesh can move on when it scoops flesh.
    """
    bowl, pose = bowl_at(cutting, spoon, np.array([0.0, SpoonWay().side, 0.085]))
    flesh = frozenset({cutting.FruitMaterial.FLESH})
    assert not fruit.bowl_blocked(bowl, pose, flesh)


def test_bowl_reaching_the_skin_is_blocked_when_it_scoops_flesh(cutting, fruit, spoon):
    """
    A bowl reaching into the skin is stopped when it scoops flesh only, and not when
    it scoops the skin too.
    """
    skin = fruit.pieces[0].part_of(cutting.FruitMaterial.SKIN).solid
    vertices = skin.to_mesh().vert_properties[:, :3]
    at_height = vertices[np.abs(vertices[:, 2] - 0.085) < 0.002]
    outermost = at_height[np.argmin(at_height[:, 1])]
    bowl, pose = bowl_at(cutting, spoon, outermost)
    flesh = frozenset({cutting.FruitMaterial.FLESH})
    assert fruit.bowl_blocked(bowl, pose, flesh)
    assert not fruit.bowl_blocked(bowl, pose)


def test_bowl_reaching_the_pit_is_blocked_whatever_it_scoops(cutting, fruit, spoon):
    """
    A bowl reaching into the pit is stopped even when it scoops every material, as no
    knife cuts the pit.
    """
    pit = fruit.pieces[0].part_of(cutting.FruitMaterial.PIT)
    bowl, pose = bowl_at(cutting, spoon, pit.mass_distribution().centre)
    assert fruit.bowl_blocked(bowl, pose, frozenset(cutting.FruitMaterial))


def test_spoon_slides_along_the_skin_in_its_way(cutting, scooping_demo):
    """
    A spoon led deeper than the flesh is thick slides along the skin instead of being
    stopped, and scoops without cutting into the skin.
    """
    demonstration = scooping_demo.AvocadoScoopingDemonstration(
        headless=True, random_pit=None, scoop_rise=-0.008
    )
    skin = cutting.FruitMaterial.SKIN
    whole_skin = demonstration.fruit.pieces[0].part_of(skin).solid.volume()
    outcome = demonstration.run()
    skin_after = sum(
        piece.part_of(skin).solid.volume()
        for piece in demonstration.fruit.pieces
        if skin in piece.materials()
    )
    assert not outcome.spoon_stopped
    assert outcome.scooped_flesh > 0
    assert skin_after == pytest.approx(whole_skin, rel=1e-6)


# %% a random pit


@pytest.fixture
def random_pit(cutting: ModuleType) -> ModuleType:
    """
    The ``random_pit`` module of the avocado demos, imported while :func:`cutting`
    makes their folder importable.
    """
    return importlib.import_module("random_pit")


def fruit_with_pit(
    cutting: ModuleType, knife: KnifeInScene, pit: Any, seed: int
) -> Any:
    """
    :param cutting: The ``cutting`` module.
    :param knife: The knife to put next to the fruit.
    :param pit: The random pit to give the fruit.
    :param seed: The seed of the random pit.
    :return: The whole solid avocado of :func:`floating_fruit`, with the random pit.
    """
    spec = mujoco.MjSpec.from_file(str(DemoPaths().solid_avocado()))
    spec.option.gravity = [0, 0, 0]
    knife.add_to(spec)
    pit.apply_to(spec, "avocado", seed)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return cutting.CuttableFruit(spec, model, data, "avocado")


def pit_and_flesh(cutting: ModuleType, fruit: Any) -> tuple[Any, Any]:
    """
    :param cutting: The ``cutting`` module.
    :param fruit: A whole fruit.
    :return: Its pit and its flesh.
    """
    whole = fruit.pieces[0]
    return (
        whole.part_of(cutting.FruitMaterial.PIT),
        whole.part_of(cutting.FruitMaterial.FLESH),
    )


def test_random_pit_lies_in_a_cavity_of_the_flesh(cutting, knife, random_pit, fruit):
    """
    A random pit lies elsewhere than the avocado's own, and the flesh leaves room for
    it: the two do not overlap.
    """
    own_pit, _ = pit_and_flesh(cutting, fruit)
    pit, flesh = pit_and_flesh(
        cutting, fruit_with_pit(cutting, knife, random_pit.RandomPit(), 3)
    )
    assert (pit.solid ^ flesh.solid).volume() == pytest.approx(0.0, abs=1e-12)
    assert not np.allclose(
        pit.mass_distribution().centre, own_pit.mass_distribution().centre
    )


def test_same_seed_gives_the_same_pit(cutting, knife, random_pit):
    """
    The same seed gives the same pit, another seed another one.
    """
    pit = random_pit.RandomPit()
    centres = [
        pit_and_flesh(cutting, fruit_with_pit(cutting, knife, pit, seed))[0]
        .mass_distribution()
        .centre
        for seed in (5, 5, 6)
    ]
    assert centres[0] == pytest.approx(centres[1])
    assert not np.allclose(centres[0], centres[2])


def test_pit_too_large_for_the_flesh_is_refused(cutting, knife, random_pit):
    """
    A pit that cannot fit into the flesh raises the fitting exception.
    """
    exceptions = sys.modules["exceptions"]
    too_large = random_pit.RandomPit(size_range=(1.6, 1.6), attempts=2)
    with pytest.raises(exceptions.PitDoesNotFit):
        fruit_with_pit(cutting, knife, too_large, 0)


def test_points_next_to_the_pit_need_not_be_passed(cutting, fruit):
    """
    Of a section through the pit, the blade need not pass the points next to the pit,
    which it keeps clear of, but has to pass those farther away.
    """
    plane = cutting.CuttingPlane([0.003, 0.002, 0.0485], [1.0, 0.0, 0.0])
    whole = fruit.pieces[0]
    samples = whole.section(
        plane, {cutting.FruitMaterial.FLESH, cutting.FruitMaterial.SKIN}
    ).samples(fruit.sample_spacing)
    reachable = fruit.reachable(whole, plane, samples)
    pit_corners = whole.section(plane, {cutting.FruitMaterial.PIT}).outline_points()
    distances = np.linalg.norm(
        reachable[:, None, :] - pit_corners[None, :, :], axis=-1
    ).min(axis=1)
    assert 0 < len(reachable) < len(samples)
    assert distances.min() >= fruit.uncuttable_margin


# %% the pieces in a semantic digital twin


@pytest.fixture
def fruit_world(cutting: ModuleType) -> ModuleType:
    """
    The ``fruit_world`` module of the avocado demos, imported while :func:`cutting`
    makes their folder importable.
    """
    return importlib.import_module("fruit_world")


def move_every_piece(fruit: Any) -> None:
    """
    :param fruit: A fruit whose pieces are all free; each is moved and turned apart.
    """
    model = fruit.model
    for number, piece in enumerate(fruit.pieces):
        address = model.jnt_qposadr[model.body_jntadr[fruit.body_id(piece)]]
        quaternion = np.array([1.0, 0.2 * number, -0.3, 0.1])
        fruit.data.qpos[address : address + 7] = [
            0.1 * number,
            -0.05,
            0.3,
            *(quaternion / np.linalg.norm(quaternion)),
        ]
    mujoco.mj_forward(model, fruit.data)


def test_world_holds_a_body_for_each_piece_and_its_parts(
    fruit_world, fruit, slanted_plane
):
    """
    Each piece is a body of the world holding a body for each of its parts; the body of
    a piece that was cut leaves the world with its parts.
    """
    mirror = fruit_world.FruitWorld()
    mirror.update(fruit)
    fruit.cut(slanted_plane)
    mirror.update(fruit)
    for piece in fruit.pieces:
        body = mirror.body_of(piece)
        assert body.name.name == piece.body_name
        assert {part.name.name for part in body.child_kinematic_structure_entities} == {
            f"{piece.body_name}_{part.material}" for part in piece.parts
        }
    assert len(mirror.world.bodies) == 1 + sum(
        1 + len(piece.parts) for piece in fruit.pieces
    )


def test_piece_bodies_are_where_the_pieces_are(fruit_world, fruit, slanted_plane):
    """
    After the pieces have moved, each piece's body lies where the piece does.
    """
    fruit.cut(slanted_plane)
    mirror = fruit_world.FruitWorld()
    mirror.update(fruit)
    move_every_piece(fruit)
    mirror.update(fruit)
    for piece in fruit.pieces:
        pose = fruit.pose_of(piece)
        transform = mirror.world.compute_forward_kinematics_np(
            mirror.world.root, mirror.body_of(piece)
        )
        assert transform[:3, 3] == pytest.approx(pose.position)
        assert transform[:3, :3] == pytest.approx(pose.rotation())


def test_part_bodies_look_like_the_parts(fruit_world, fruit, slanted_plane):
    """
    The body of each part has the part's shape.
    """
    fruit.cut(slanted_plane)
    mirror = fruit_world.FruitWorld()
    mirror.update(fruit)
    for piece in fruit.pieces:
        bodies = {
            body.name.name: body
            for body in mirror.body_of(piece).child_kinematic_structure_entities
        }
        for part in piece.parts:
            shapes = bodies[f"{piece.body_name}_{part.material}"].visual.shapes
            assert len(shapes) == 1
            assert shapes[0].mesh.volume == pytest.approx(part.solid.volume(), rel=1e-6)


def test_piece_bodies_collide_as_the_pieces_do_in_mujoco(
    fruit_world, fruit, slanted_plane
):
    """
    A piece's body collides through the convex hulls of the geoms the piece collides
    through in MuJoCo.
    """
    fruit.cut(slanted_plane)
    mirror = fruit_world.FruitWorld()
    mirror.update(fruit)
    model = fruit.model
    for piece in fruit.pieces:
        body_id = fruit.body_id(piece)
        geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == body_id
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ]
        volumes = sorted(
            shape.mesh.volume for shape in mirror.body_of(piece).collision.shapes
        )
        expected = sorted(
            fruit.mesh_in_body_frame(geom_id).convex_hull.volume for geom_id in geom_ids
        )
        assert volumes == pytest.approx(expected, rel=1e-6)


def test_demonstration_keeps_its_world_in_step_with_the_simulation(scooping_demo):
    """
    Once the update interval has passed, a demonstration's world has a body where each
    piece is.
    """
    demonstration = scooping_demo.AvocadoScoopingDemonstration(
        headless=True, random_pit=None
    )
    fruit = demonstration.fruit
    fruit.release(fruit.pieces[0])
    timestep = fruit.model.opt.timestep
    for _ in range(round(demonstration.world_update_interval / timestep) + 1):
        demonstration.step()
    mirror = demonstration.fruit_world
    (piece,) = fruit.pieces
    transform = mirror.world.compute_forward_kinematics_np(
        mirror.world.root, mirror.body_of(piece)
    )
    assert transform[:3, 3] == pytest.approx(fruit.pose_of(piece).position, abs=1e-3)


def test_piece_bodies_weigh_what_the_pieces_weigh(fruit_world, fruit, slanted_plane):
    """
    A piece's body has the piece's mass, centre of mass and inertia.
    """
    fruit.cut(slanted_plane)
    mirror = fruit_world.FruitWorld()
    mirror.update(fruit)
    for piece in fruit.pieces:
        inertial = mirror.body_of(piece).inertial
        distribution = piece.mass_distribution()
        assert inertial.mass == pytest.approx(distribution.mass)
        assert inertial.center_of_mass.to_np()[:3] == pytest.approx(distribution.centre)
        assert inertial.inertia.data == pytest.approx(distribution.inertia)


# %% the pieces as a USD stage


@dataclass
class StageOfCutFruit:
    """
    A fruit cut in two, its pieces moved apart, and the USD stage written of its world.
    """

    fruit: Any
    """
    The fruit.
    """

    mirror: Any
    """
    The world of its pieces.
    """

    stage: Any
    """
    The stage, read back from its file.
    """

    floor_height: float
    """
    The height of the floor the stage was written with, in metres.
    """


@pytest.fixture
def stage_of_cut_fruit(
    cutting: ModuleType, fruit: Any, slanted_plane: Any, tmp_path: Path
) -> StageOfCutFruit:
    """
    The stage of :class:`StageOfCutFruit`; skipped where USD is not installed.
    """
    usd = pytest.importorskip("pxr.Usd")
    fruit_world = importlib.import_module("fruit_world")
    fruit_stage = importlib.import_module("fruit_stage")
    fruit.cut(slanted_plane)
    move_every_piece(fruit)
    mirror = fruit_world.FruitWorld()
    mirror.update(fruit)
    path = tmp_path / "pieces.usda"
    floor_height = 0.02
    fruit_stage.FruitStage(mirror, floor_height).write(path)
    return StageOfCutFruit(fruit, mirror, usd.Stage.Open(str(path)), floor_height)


def test_stage_has_a_rigid_body_where_each_piece_is(stage_of_cut_fruit):
    """
    Each piece is a rigid body of the stage, where the piece is, weighing what it weighs.
    """
    pxr = pytest.importorskip("pxr")
    fruit = stage_of_cut_fruit.fruit
    for piece in fruit.pieces:
        prim = stage_of_cut_fruit.stage.GetPrimAtPath(f"/World/{piece.body_name}")
        assert prim.HasAPI(pxr.UsdPhysics.RigidBodyAPI)
        transform = pxr.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            pxr.Usd.TimeCode.Default()
        )
        pose = fruit.pose_of(piece)
        assert np.array(transform.ExtractTranslation()) == pytest.approx(
            pose.position, abs=1e-6
        )
        assert np.array(transform.ExtractRotationMatrix()).T == pytest.approx(
            pose.rotation(), abs=1e-5
        )
        mass = pxr.UsdPhysics.MassAPI(prim)
        distribution = piece.mass_distribution()
        assert mass.GetMassAttr().Get() == pytest.approx(distribution.mass, rel=1e-6)
        assert np.array(mass.GetCenterOfMassAttr().Get()) == pytest.approx(
            distribution.centre, abs=1e-6
        )


def test_parts_are_meshes_seen_but_not_collided_with(stage_of_cut_fruit):
    """
    Each part is a mesh of its piece's rigid body with the part's surface, which is seen
    and takes no part in collisions.
    """
    pxr = pytest.importorskip("pxr")
    stage = stage_of_cut_fruit.stage
    for piece in stage_of_cut_fruit.fruit.pieces:
        for part in piece.parts:
            name = f"{piece.body_name}_{part.material}"
            prim = stage.GetPrimAtPath(f"/World/{piece.body_name}/{name}")
            surface = part.triangle_mesh_of(part.solid)
            assert len(pxr.UsdGeom.Mesh(prim).GetPointsAttr().Get()) == len(
                surface.vertices
            )
            assert not prim.HasAPI(pxr.UsdPhysics.CollisionAPI)


def test_pieces_collide_through_convex_hulls(stage_of_cut_fruit):
    """
    Each piece collides through one convex hull per collision shape of its body, which
    is not seen.
    """
    pxr = pytest.importorskip("pxr")
    stage = stage_of_cut_fruit.stage
    for piece in stage_of_cut_fruit.fruit.pieces:
        body = stage_of_cut_fruit.mirror.body_of(piece)
        colliders = [
            prim
            for prim in stage.GetPrimAtPath(
                f"/World/{piece.body_name}"
            ).GetAllChildren()
            if prim.HasAPI(pxr.UsdPhysics.CollisionAPI)
        ]
        assert len(colliders) == len(body.collision.shapes)
        for prim in colliders:
            assert (
                pxr.UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                == pxr.UsdPhysics.Tokens.convexHull
            )
            assert (
                pxr.UsdGeom.Imageable(prim).GetPurposeAttr().Get()
                == pxr.UsdGeom.Tokens.guide
            )


def test_stage_is_simulated_on_a_floor_under_gravity(stage_of_cut_fruit):
    """
    The stage has a physics scene with gravity and a floor at the given height, in
    metres with z up.
    """
    pxr = pytest.importorskip("pxr")
    stage = stage_of_cut_fruit.stage
    assert pxr.UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert pxr.UsdGeom.GetStageUpAxis(stage) == pxr.UsdGeom.Tokens.z
    scenes = [prim for prim in stage.Traverse() if prim.IsA(pxr.UsdPhysics.Scene)]
    assert len(scenes) == 1
    floor = stage.GetPrimAtPath("/World/floor")
    assert floor.HasAPI(pxr.UsdPhysics.CollisionAPI)
    assert not floor.HasAPI(pxr.UsdPhysics.RigidBodyAPI)
    height = (
        pxr.UsdGeom.Xformable(floor)
        .ComputeLocalToWorldTransform(pxr.Usd.TimeCode.Default())
        .ExtractTranslation()[2]
    )
    assert height == pytest.approx(stage_of_cut_fruit.floor_height)
