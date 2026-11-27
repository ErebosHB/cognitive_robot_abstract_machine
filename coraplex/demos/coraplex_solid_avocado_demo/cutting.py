"""
Cutting a fruit made of solid parts - skin, flesh and pit - exactly where a knife goes
through it. Each piece cut off is a body of its own; a piece something holds is fixed in
place, a spoon's bowl scoops out what it held on its way, and a piece can be cut into
segments joined by hinges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import manifold3d
import mujoco
import numpy as np
import trimesh

from adhesion import Adhesion
from exceptions import FruitPartNotFound, FruitPartNotSolid, PiecesNotHinged
from fruit_material import FruitMaterial
from texture import FruitTexture

# %% planes and knives


@dataclass
class CuttingPlane:
    """
    The plane a knife cuts along.
    """

    point: np.ndarray
    """
    A point of the plane, in metres.
    """

    normal: np.ndarray
    """
    The plane's normal.
    """

    def __post_init__(self) -> None:
        """
        Make :attr:`normal` a unit vector.
        """
        self.point = np.asarray(self.point, dtype=float)
        self.normal = np.asarray(self.normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(self.normal)

    def lies_near(self, *, other: CuttingPlane, tolerance: float) -> bool:
        """
        :param other: Another plane in the same frame.
        :param tolerance: How far, in metres and in radians, it may lie from this one.
        :return: Whether it is turned and shifted from this one by no more than that.
        """
        tilt = float(np.linalg.norm(other.normal - self.normal))
        shift = abs(float((other.point - self.point) @ self.normal))
        return tilt <= tolerance and shift <= tolerance

    def seen_from(self, pose: BodyPose) -> CuttingPlane:
        """
        :param pose: The pose of a body in the frame this plane is given in.
        :return: This plane in the body's frame.
        """
        rotation = pose.rotation()
        return CuttingPlane(
            rotation.T @ (self.point - pose.position), rotation.T @ self.normal
        )


@dataclass
class BodyPose:
    """
    Where a free body is and how it moves.
    """

    position: np.ndarray
    """
    The position of the body's frame in the world, in metres.
    """

    quaternion: np.ndarray
    """
    The orientation of the body's frame in the world, scalar first.
    """

    velocity: np.ndarray
    """
    The velocity of the body's free joint: linear in world axes, then angular in the
    body's axes.
    """

    def rotation(self) -> np.ndarray:
        """
        :return: The rotation matrix of :attr:`quaternion`.
        """
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, self.quaternion)
        return matrix.reshape(3, 3)

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """
        :param points: Points in the body's frame, one row each.
        :return: The points in the world.
        """
        return points @ self.rotation().T + self.position

    def placement(self) -> np.ndarray:
        """
        :return: The 3x4 matrix that moves solids from the body's frame into the
            world.
        """
        return np.column_stack([self.rotation(), self.position])

    def removal(self) -> np.ndarray:
        """
        :return: The 3x4 matrix that moves solids from the world into the body's
            frame.
        """
        rotation = self.rotation()
        return np.column_stack([rotation.T, -rotation.T @ self.position])

    def placement_in(self, *, frame: BodyPose) -> np.ndarray:
        """
        :param frame: The pose of another body.
        :return: The 3x4 matrix that moves solids from this body's frame into the
            other body's frame.
        """
        to_frame = frame.removal()
        return np.column_stack(
            [
                to_frame[:, :3] @ self.rotation(),
                to_frame[:, :3] @ self.position + to_frame[:, 3],
            ]
        )


@dataclass
class KnifeBlade:
    """
    The blade of a knife in a MuJoCo scene, placed by a site in the middle of its
    cutting edge: the site's x axis runs along the edge, its y axis from the edge into
    the blade, and its z axis across the blade.
    """

    site_name: str
    """
    The name of the site on the cutting edge.
    """

    edge_length: float
    """
    The length of the cutting edge in metres.
    """

    def site_id(self, model: mujoco.MjModel) -> int:
        """
        :param model: The model holding the knife.
        :return: The id of the blade's site.
        :raises FruitPartNotFound: If the model has no such site.
        """
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        if site_id < 0:
            raise FruitPartNotFound(self.site_name)
        return site_id

    def plane(self, model: mujoco.MjModel, data: mujoco.MjData) -> CuttingPlane:
        """
        :param model: The model holding the knife.
        :param data: Its state.
        :return: The plane the blade lies in, in the world.
        """
        site_id = self.site_id(model)
        across_blade = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        return CuttingPlane(data.site_xpos[site_id].copy(), across_blade.copy())

    def offsets_of(
        self, model: mujoco.MjModel, data: mujoco.MjData, points: np.ndarray
    ) -> np.ndarray:
        """
        :param model: The model holding the knife.
        :param data: Its state.
        :param points: Points in the world, one row each.
        :return: Each point's offset from the middle of the cutting edge, along the
            edge, into the blade and across it, in metres.
        """
        site_id = self.site_id(model)
        return (points - data.site_xpos[site_id]) @ data.site_xmat[site_id].reshape(
            3, 3
        )

    def passed(
        self, model: mujoco.MjModel, data: mujoco.MjData, points: np.ndarray
    ) -> np.ndarray:
        """
        :param model: The model holding the knife.
        :param data: Its state.
        :param points: Points of the blade's plane in the world, one row each.
        :return: For each point, whether the cutting edge has gone past it: it lies
            behind the edge, between the edge's ends.
        """
        offsets = self.offsets_of(model, data, points)
        return (offsets[:, 1] >= 0) & (np.abs(offsets[:, 0]) <= self.edge_length / 2)

    def room_ahead(
        self, model: mujoco.MjModel, data: mujoco.MjData, points: np.ndarray
    ) -> float:
        """
        :param model: The model holding the knife.
        :param data: Its state.
        :param points: Points of the blade's plane in the world, one row each.
        :return: How far in metres the cutting edge can still move forward before it
            reaches one of the points; infinity if none lies ahead of it.
        """
        offsets = self.offsets_of(model, data, points)
        ahead = (offsets[:, 1] < 0) & (np.abs(offsets[:, 0]) <= self.edge_length / 2)
        if not ahead.any():
            return float("inf")
        return float(-offsets[ahead, 1].max())


# %% sections and incisions


@dataclass
class SpoonBowl:
    """
    The bowl of a spoon on a body of a MuJoCo model: an ellipsoid round the body's
    frame, its back facing the frame's z axis and its hollow the other way.
    """

    body_name: str
    """
    The name of the body.
    """

    half_size: tuple[float, float, float]
    """
    Half the length, width and thickness of the bowl, along the body's x, y and z
    axes, in metres.
    """

    reach: float = 0.02
    """
    How far beyond its back, on its hollow side, the bowl holds what it scoops, in
    metres.
    """

    def pose(self, model: mujoco.MjModel, data: mujoco.MjData) -> BodyPose:
        """
        :param model: The compiled scene.
        :param data: Its simulation state.
        :return: Where the bowl's body is.
        """
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        return BodyPose(
            data.xpos[body_id].copy(), data.xquat[body_id].copy(), np.zeros(6)
        )

    def solid(self) -> manifold3d.Manifold:
        """
        :return: The bowl, in its body's frame.
        """
        return manifold3d.Manifold.sphere(1.0, 32).scale(list(self.half_size))

    def holding_space(self) -> manifold3d.Manifold:
        """
        :return: The bowl and what lies on its hollow side as far as :attr:`reach`,
            in its body's frame.
        """
        bowl = self.solid()
        return manifold3d.Manifold.batch_hull(
            [bowl, bowl.translate([0.0, 0.0, -self.reach])]
        )


@dataclass
class Sweep:
    """
    The space a spoon's bowl has held on its way through a piece since it went in.
    """

    space: manifold3d.Manifold
    """
    That space, in the frame of the piece's body.
    """

    last_space: manifold3d.Manifold
    """
    The space the bowl held where it was last added to :attr:`space`, in the frame
    of the piece's body.
    """

    last_centre: np.ndarray
    """
    Where the middle of the bowl was then, in the frame of the piece's body, in
    metres.
    """

    filled: bool = False
    """
    Whether the bowl has been full of the material it scoops.
    """


@dataclass
class PlaneSection:
    """
    The area a plane cuts out of solids, as the closed loops of its outline.
    """

    plane: CuttingPlane
    """
    The plane, in the frame of the solids.
    """

    in_plane_axes: np.ndarray
    """
    Two orthonormal directions in the plane, one row each, that the loops' coordinates
    are measured along from :attr:`CuttingPlane.point`.
    """

    loops: list[np.ndarray]
    """
    The loops of the outline, one point per row in metres along :attr:`in_plane_axes`.
    """

    @classmethod
    def through(
        cls, solids: list[manifold3d.Manifold], plane: CuttingPlane
    ) -> PlaneSection:
        """
        :param solids: Solids in one frame.
        :param plane: A plane in that frame.
        :return: The area the plane cuts out of the solids together.
        """
        helper = np.eye(3)[int(np.argmin(np.abs(plane.normal)))]
        first_axis = np.cross(plane.normal, helper)
        first_axis /= np.linalg.norm(first_axis)
        axes = np.array([first_axis, np.cross(plane.normal, first_axis), plane.normal])
        # Moves the plane onto z = 0, where manifold3d slices.
        onto_plane = np.column_stack([axes, -axes @ plane.point])
        joined = manifold3d.Manifold.batch_boolean(solids, manifold3d.OpType.Add)
        loops = joined.transform(onto_plane).slice(0.0).to_polygons()
        return cls(plane, axes[:2], [np.asarray(loop, dtype=float) for loop in loops])

    def to_solid_frame(self, coordinates: np.ndarray) -> np.ndarray:
        """
        :param coordinates: Points of the plane in metres along :attr:`in_plane_axes`,
            one row each.
        :return: The points in the frame of the solids.
        """
        return self.plane.point + coordinates @ self.in_plane_axes

    def outline_points(self) -> np.ndarray:
        """
        :return: The corners of the outline's loops in the frame of the solids, one row
            each.
        """
        return self.to_solid_frame(np.vstack([np.empty((0, 2)), *self.loops]))

    def contains(self, coordinates: np.ndarray) -> np.ndarray:
        """
        :param coordinates: Points of the plane in metres along :attr:`in_plane_axes`,
            one row each.
        :return: For each point, whether it lies inside the area.
        """
        inside = np.zeros(len(coordinates), dtype=bool)
        across, up = coordinates[:, :1], coordinates[:, 1:]
        for loop in self.loops:
            start, end = loop, np.roll(loop, -1, axis=0)
            rise = end[:, 1] - start[:, 1]
            spans = (start[:, 1] > up) != (end[:, 1] > up)
            crossing = start[:, 0] + (up - start[:, 1]) * (end[:, 0] - start[:, 0]) / (
                np.where(rise == 0, 1.0, rise)
            )
            # A point is inside when a ray from it crosses the outline an odd number
            # of times.
            inside ^= np.count_nonzero(spans & (across < crossing), axis=1) % 2 == 1
        return inside

    def samples(self, spacing: float) -> np.ndarray:
        """
        :param spacing: The distance in metres between neighbouring points of the grid.
        :return: The corners of the outline and the points of a square grid inside the
            area, in the frame of the solids, one row each.
        """
        corners = np.vstack([np.empty((0, 2)), *self.loops])
        if not len(corners):
            return np.empty((0, 3))
        lowest, highest = corners.min(axis=0), corners.max(axis=0)
        grid = np.stack(
            np.meshgrid(
                np.arange(lowest[0], highest[0], spacing),
                np.arange(lowest[1], highest[1], spacing),
            ),
            axis=-1,
        ).reshape(-1, 2)
        return self.to_solid_frame(np.vstack([corners, grid[self.contains(grid)]]))


@dataclass
class Incision:
    """
    A cut a blade has begun in one piece along one plane: which points of the piece's
    section the cutting edge has gone past so far.
    """

    plane: CuttingPlane
    """
    The plane of the cut, in the frame of the piece's body.
    """

    materials: frozenset[FruitMaterial]
    """
    The materials the cut goes through.
    """

    samples: np.ndarray
    """
    Points of the area of the piece the cut has to go through, in the frame of the
    piece's body, one row each.
    """

    passed: np.ndarray = field(init=False)
    """
    For each of :attr:`samples`, whether the cutting edge has gone past it.
    """

    def __post_init__(self) -> None:
        """
        Start with no point passed.
        """
        self.passed = np.zeros(len(self.samples), dtype=bool)

    def follows(self, plane: CuttingPlane, tolerance: float) -> bool:
        """
        :param plane: A plane in the frame of the piece's body.
        :param tolerance: How far, in metres and in radians, the plane may lie from the
            plane of this cut.
        :return: Whether a blade in that plane carries on with this cut.
        """
        return self.plane.lies_near(other=plane, tolerance=tolerance)


# %% parts and pieces


@dataclass
class MassDistribution:
    """
    The mass of a rigid body and how it is spread.
    """

    mass: float
    """
    The mass in kilograms.
    """

    centre: np.ndarray
    """
    The centre of mass in the body's frame, in metres.
    """

    inertia: np.ndarray
    """
    The inertia matrix about :attr:`centre`, in the body's axes.
    """


@dataclass(kw_only=True)
class ContactPatch:
    """
    Where two convex geoms press together: one patch, however many contacts MuJoCo
    reports in it.
    """

    force: float = 0.0
    """
    The normal force the geoms press together with over the patch, in newtons.
    """

    weighted_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """
    The sum of the position of each contact in the patch, in the world, in metres,
    times its normal force.
    """

    def add_contact(self, *, force: float, position: np.ndarray) -> None:
        """
        :param force: The normal force of a contact in the patch, in newtons.
        :param position: Where the contact is, in the world, in metres.
        """
        self.force += force
        self.weighted_position += force * position

    @property
    def centre(self) -> np.ndarray:
        """
        :return: Where the patch presses, in the world, in metres: the position of its
            contacts, weighted by their normal forces.
        """
        return self.weighted_position / self.force


@dataclass(kw_only=True)
class Bruise:
    """
    A place where a press has bruised a fruit's flesh.
    """

    point: np.ndarray
    """
    Where, in the frame of the fruit, in metres.
    """

    pressure: float
    """
    The contact pressure of the press, in pascals.
    """

    reach: float = 0.004
    """
    How far in metres from :attr:`point`, which lies on the surface pressed, the
    bruise reaches into the fruit.
    """

    def affects(self, *, part: FruitPart) -> bool:
        """
        :param part: A part of a piece of the fruit.
        :return: Whether the bruise reaches into the part.
        """
        reached = manifold3d.Manifold.sphere(self.reach, 12).translate(
            self.point.tolist()
        )
        return (part.solid ^ reached).volume() > 0.0


@dataclass
class FruitPart:
    """
    The part of one piece of fruit that is of one material.
    """

    material: FruitMaterial
    """
    What the part is made of.
    """

    solid: manifold3d.Manifold
    """
    The part's shape, in the frame of its piece's body.
    """

    geom_class: str
    """
    The name of the MuJoCo default class that gives the part's geom its looks, its
    density and its contact properties.
    """

    density: float
    """
    The material's density in kilograms per cubic metre.
    """

    collides_as_is: bool
    """
    Whether the part's geom collides itself, as the convex hull of its mesh, instead of
    through the convex shapes of its piece.
    """

    @staticmethod
    def solid_of(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
        """
        :param mesh: A closed triangle mesh.
        :return: The solid it encloses.
        :raises FruitPartNotSolid: If the mesh does not enclose a volume.
        """
        solid = manifold3d.Manifold(
            manifold3d.Mesh(
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.uint32),
            )
        )
        if solid.status() != manifold3d.Error.NoError or solid.is_empty():
            raise FruitPartNotSolid(len(mesh.vertices), len(mesh.faces))
        return solid

    @staticmethod
    def triangle_mesh_of(solid: manifold3d.Manifold) -> trimesh.Trimesh:
        """
        :param solid: A solid.
        :return: Its surface.
        """
        mesh = solid.to_mesh()
        return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts)

    @staticmethod
    def touching_regions(
        parts: list[FruitPart], negligible_volume: float
    ) -> list[manifold3d.Manifold]:
        """
        :param parts: Parts in one frame.
        :param negligible_volume: The volume in cubic metres below which a region is a
            numerical leftover between parts that share a surface.
        :return: The separate solids the parts form together, parts that touch joined.
        """
        joined = manifold3d.Manifold.batch_boolean(
            [part.solid for part in parts], manifold3d.OpType.Add
        )
        return [
            region
            for region in joined.decompose()
            if region.volume() > negligible_volume
        ]

    def within(self, region: manifold3d.Manifold) -> FruitPart:
        """
        :param region: A solid in the frame of the part's piece.
        :return: What of this part lies in the region.
        """
        return FruitPart(
            self.material,
            self.solid ^ region,
            self.geom_class,
            self.density,
            self.collides_as_is,
        )

    def moved_whole(self) -> FruitPart:
        """
        :return: A copy of this part for another piece.
        """
        return self.within(self.solid)

    def mass_distribution(self) -> MassDistribution:
        """
        :return: The part's mass and how it is spread, in the frame of its piece.
        """
        mesh = self.triangle_mesh_of(self.solid)
        mesh.density = self.density
        return MassDistribution(mesh.mass, mesh.center_mass, mesh.moment_inertia)


@dataclass(eq=False)
class FruitPiece:
    """
    One rigid piece of a fruit: a free body holding one part of each material it has.

    Two pieces are equal only if they are the same piece.
    """

    body_name: str
    """
    The name of the piece's body.
    """

    parts: list[FruitPart]
    """
    The parts the piece is made of.
    """

    mesh_names: list[str] = field(default_factory=list)
    """
    The names of the MuJoCo meshes made for this piece.
    """

    negligible_volume: float = 1e-9
    """
    The volume in cubic metres below which a solid is a numerical leftover.
    """

    surface_tolerance: float = 1e-5
    """
    The distance in metres within which a plane along a surface of the piece counts as
    touching the piece, not as crossing it.
    """

    outline: trimesh.Trimesh = field(init=False)
    """
    The surface of the whole piece, in the frame of its body.
    """

    distribution: MassDistribution | None = field(init=False, default=None)
    """
    The piece's mass and how it is spread, once :meth:`mass_distribution` has worked it
    out.
    """

    def __post_init__(self) -> None:
        """
        Set :attr:`outline` from :attr:`parts`.
        """
        self.outline = FruitPart.triangle_mesh_of(
            manifold3d.Manifold.compose(
                FruitPart.touching_regions(self.parts, self.negligible_volume)
            )
        )

    def materials(self) -> set[FruitMaterial]:
        """
        :return: The materials the piece has a part of.
        """
        return {part.material for part in self.parts}

    def part_of(self, material: FruitMaterial) -> FruitPart:
        """
        :param material: A material.
        :return: The piece's part of that material.
        :raises FruitPartNotFound: If the piece has none.
        """
        for part in self.parts:
            if part.material == material:
                return part
        raise FruitPartNotFound(f"{self.body_name}_{material}")

    def parts_of(self, *, materials: frozenset[FruitMaterial]) -> list[FruitPart]:
        """
        :param materials: Materials.
        :return: The piece's parts of those materials.
        """
        return [part for part in self.parts if part.material in materials]

    def volume(self) -> float:
        """
        :return: The volume of the piece's parts in cubic metres.
        """
        return sum(part.solid.volume() for part in self.parts)

    def mass_distribution(self) -> MassDistribution:
        """
        :return: The piece's mass and how it is spread, in the frame of its body.
        """
        if self.distribution is not None:
            return self.distribution
        distributions = [part.mass_distribution() for part in self.parts]
        mass = sum(distribution.mass for distribution in distributions)
        centre = (
            sum(
                distribution.mass * distribution.centre
                for distribution in distributions
            )
            / mass
        )
        inertia = np.zeros((3, 3))
        for distribution in distributions:
            offset = distribution.centre - centre
            inertia += distribution.inertia + distribution.mass * (
                offset @ offset * np.eye(3) - np.outer(offset, offset)
            )
        self.distribution = MassDistribution(mass, centre, inertia)
        return self.distribution

    def section(
        self, plane: CuttingPlane, materials: set[FruitMaterial]
    ) -> PlaneSection | None:
        """
        :param plane: A plane in the frame of the piece's body.
        :param materials: The materials whose parts the section goes through.
        :return: The area the plane cuts out of the piece's parts of those materials;
            None if it misses them or only touches them.
        """
        solids = [part.solid for part in self.parts if part.material in materials]
        if not solids:
            return None
        vertices = np.vstack(
            [solid.to_mesh().vert_properties[:, :3] for solid in solids]
        )
        heights = (vertices - plane.point) @ plane.normal
        if min(heights.max(), -heights.min()) < self.surface_tolerance:
            return None
        return PlaneSection.through(solids, plane)


# %% the fruit in a simulation


@dataclass
class CuttableFruit:
    """
    A fruit of solid parts in a MuJoCo simulation that is whole until it is cut.

    It starts as the body :attr:`body_name`. The geom named after the body and a
    :class:`FruitMaterial` is the part of that material; the other geoms are the convex
    shapes the body collides through. MuJoCo cannot split a body while it simulates, so
    a cut cuts the parts' meshes and compiles :attr:`spec` again.

    .. warning:: Every cut, release and hold replaces :attr:`model` and :attr:`data`.
        Read them from the fruit after each such call, and hand them to any viewer
        showing the old ones.
    """

    spec: mujoco.MjSpec
    """
    The scene the fruit is in; :attr:`model` was compiled from it.
    """

    model: mujoco.MjModel
    """
    The compiled scene.
    """

    data: mujoco.MjData
    """
    The simulation state of :attr:`model`.
    """

    body_name: str
    """
    The name of the body of the whole fruit, with the prefix it was attached with.
    """

    holders: list[manifold3d.Manifold] = field(default_factory=list)
    """
    Solids in the world, such as the tips of poles stuck into the fruit, that hold in
    place every piece reaching into them.
    """

    uncuttable: frozenset[FruitMaterial] = frozenset({FruitMaterial.PIT})
    """
    The materials a knife does not cut: such a part stays whole, in the piece on the
    side of the plane holding most of it.
    """

    uncuttable_margin: float = 0.0015
    """
    How close in metres to a part no knife cuts a point of a piece's section may lie
    and need not be passed by the blade for the piece to be cut.
    """

    smallest_piece_volume: float = 1e-7
    """
    The volume in cubic metres below which nothing is cut off: a cut that would leave a
    smaller piece of some body does not cut that body.
    """

    negligible_volume: float = 1e-9
    """
    The volume in cubic metres below which a solid is a numerical leftover of a cut and
    is dropped.
    """

    refinements: int = 3
    """
    How many times at most the region of a collision shape is divided again where the
    shape would reach into material of the fruit that is not its piece's.
    """

    sample_spacing: float = 0.002
    """
    The distance in metres between the points of a piece's section that a blade has to
    go past to cut the piece.
    """

    incision_tolerance: float = 2e-3
    """
    How far, in metres and in radians, a blade's plane may drift from the plane of the
    cut it has begun in a piece and still carry on with it.
    """

    sweep_spacing: float = 0.002
    """
    How far in metres a spoon's bowl moves between the places its way through a piece
    is taken at.
    """

    full_bowl: float = 0.9
    """
    The share of its volume the material it scoops fills a spoon's bowl to at least
    once it is in the piece.
    """

    emerging_bowl: float = 0.5
    """
    The share of its volume the material it scoops fills a full spoon's bowl to at
    most once the bowl comes out again, at which the scoop comes away.
    """

    sweeps: dict[str, Sweep] = field(init=False, default_factory=dict)
    """
    The way a spoon's bowl has taken through each piece, by the name of the piece's
    body.
    """

    hinge_time_constant: float = 0.004
    """
    How fast in seconds a hinge between pieces pulls them back together where they
    drift apart; MuJoCo's time constant of the hinge's constraints.
    """

    texture: FruitTexture | None = None
    """
    How firm the fruit is and how its material behaves; None for a fruit whose pit
    sticks to nothing and whose flesh bruises nowhere.
    """

    touching_distance: float = 0.001
    """
    How close in metres the parts of two pieces have to lie to touch, so that a piece
    sticking to one of them sticks to the other.
    """

    contact_impedance: float = 0.9
    """
    The share of a light press a contact of a fruit with a :attr:`texture` resists at
    once; the rest it resists as it sinks in.
    """

    crushed_impedance: float = 0.99
    """
    The share of a press a contact of a fruit with a :attr:`texture` resists at once
    once the press goes as deep as the standard press, where the fruit stiffens.
    """

    fastest_contact_steps: int = 2
    """
    How many simulation steps a contact takes at least to resolve, the least MuJoCo
    keeps stable.
    """

    contact_radius: float = 0.03
    """
    The radius of curvature in metres the fruit's surface is taken to have where it is
    pressed, which sets how large the area of a contact is for its force.
    """

    sections: dict[tuple[str, FruitMaterial], PlaneSection | None] = field(
        init=False, default_factory=dict
    )
    """
    The section through the part of each material of each piece in the plane a blade
    lay in when it last asked which material it is in, by the name of the piece's body
    and the material; None where the plane missed the part.
    """

    adhesions: list[Adhesion] = field(init=False, default_factory=list)
    """
    The pieces that stick together now.
    """

    scoops: list[str] = field(init=False, default_factory=list)
    """
    The names of the bodies of the pieces a spoon's bowl has scooped out, which take
    in the room the bowl swept and so lie in the bowl.
    """

    bruises: list[Bruise] = field(init=False, default_factory=list)
    """
    Where the fruit's flesh has been bruised so far.
    """

    incisions: dict[str, Incision] = field(init=False, default_factory=dict)
    """
    The cut a blade has begun in each piece, by the name of the piece's body.
    """

    material: manifold3d.Manifold = field(init=False)
    """
    Everything the whole fruit is made of, in the fruit's frame, which every piece
    keeps.
    """

    core: np.ndarray = field(init=False)
    """
    The point in the fruit's frame the regions of collision shapes spread out from: the
    middle of the parts no knife cuts, around which the fruit is hollow without them.
    """

    reach: float = field(init=False)
    """
    A distance in metres from :attr:`core` beyond which there is no fruit.
    """

    pieces: list[FruitPiece] = field(init=False)
    """
    The pieces the fruit is in now.
    """

    collision_class: str = field(init=False)
    """
    The name of the MuJoCo default class of the convex shapes pieces collide through.
    """

    pieces_made: int = field(init=False, default=0)
    """
    How many pieces cuts have made so far, which numbers the next one.
    """

    def __post_init__(self) -> None:
        """
        Read the whole fruit, the first of :attr:`pieces`, from :attr:`model`.

        :raises FruitPartNotFound: If the body, its parts or its collision shapes are
            missing.
        """
        body = self.spec.body(self.body_name)
        if body is None:
            raise FruitPartNotFound(self.body_name)
        part_names = {
            f"{self.body_name}_{material}": material for material in FruitMaterial
        }
        parts, mesh_names, collision_classes = [], set(), []
        for geom in body.geoms:
            mesh_names.add(geom.meshname)
            if geom.name not in part_names:
                collision_classes.append(geom.classname.name)
                continue
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom.name)
            parts.append(
                FruitPart(
                    material=part_names[geom.name],
                    solid=FruitPart.solid_of(self.mesh_in_body_frame(geom_id)),
                    geom_class=geom.classname.name,
                    density=geom.classname.geom.density,
                    collides_as_is=bool(self.model.geom_contype[geom_id]),
                )
            )
        if not (parts and collision_classes):
            raise FruitPartNotFound(f"{self.body_name}_{FruitMaterial.FLESH}")
        self.collision_class = collision_classes[0]
        self.pieces = [
            FruitPiece(
                self.body_name, parts, sorted(mesh_names), self.negligible_volume
            )
        ]
        self.material = manifold3d.Manifold.compose(
            FruitPart.touching_regions(parts, self.negligible_volume)
        )
        lowest, highest = self.pieces[0].outline.bounds
        cores = [part for part in parts if part.material in self.uncuttable]
        core_bounds = [
            FruitPart.triangle_mesh_of(core.solid).bounds for core in cores
        ] or [np.array([lowest, highest])]
        self.core = np.mean(core_bounds, axis=(0, 1))
        self.reach = 2 * float(np.linalg.norm(highest - lowest))
        mujoco.mj_forward(self.model, self.data)
        if self.reaches_holder(self.pieces[0], self.pose_of(self.pieces[0])):
            for joint in body.joints:
                self.spec.delete(joint)
            self.model, self.data = self.spec.recompile(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        self.soften_contacts()

    def soften_contacts(self) -> None:
        """
        Let the convex shapes the pieces collide through give under a press as the
        whole fruit does, as soft as :attr:`texture` makes it; nothing changes without
        a texture.
        """
        if self.texture is None:
            return
        time_constant = self.contact_time_constant(piece=self.pieces[0])
        geoms = [self.spec.find_default(self.collision_class).geom] + [
            geom
            for geom in self.spec.body(self.body_name).geoms
            if geom.classname.name == self.collision_class
        ]
        for geom in geoms:
            geom.solref = [time_constant, 1.0]
            geom.solimp[:3] = [
                self.contact_impedance,
                self.crushed_impedance,
                self.texture.texture.press_depth,
            ]
        self.model, self.data = self.spec.recompile(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def contact_time_constant(self, *, piece: FruitPiece) -> float:
        """
        :param piece: A piece of this fruit, which has a :attr:`texture`.
        :return: How fast in seconds the contacts of the piece's convex shapes push it
            back out, so that a steady press sinks into it as far per newton as the
            standard press does into the whole fruit, however light the piece; no faster
            than :attr:`fastest_contact_steps` allow.
        """
        return max(
            self.texture.contact_time_constant(
                mass=piece.mass_distribution().mass,
                impedance=self.contact_impedance,
            ),
            self.fastest_contact_steps * self.model.opt.timestep,
        )

    def mesh_in_body_frame(self, geom_id: int) -> trimesh.Trimesh:
        """
        :param geom_id: The id of a mesh geom of :attr:`model`.
        :return: The geom's mesh in the frame of its body.
        """
        model = self.model
        mesh_id = model.geom_dataid[geom_id]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh_id] :][
            : model.mesh_vertnum[mesh_id]
        ]
        faces = model.mesh_face[model.mesh_faceadr[mesh_id] :][
            : model.mesh_facenum[mesh_id]
        ]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        return trimesh.Trimesh(
            vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom_id], faces
        )

    # %% state

    def body_id(self, piece: FruitPiece) -> int:
        """
        :param piece: A piece of this fruit.
        :return: The id of its body in :attr:`model`.
        """
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, piece.body_name)

    def pose_of(self, piece: FruitPiece) -> BodyPose:
        """
        :param piece: A piece of this fruit.
        :return: Where its body is and how it moves.
        """
        body_id = self.body_id(piece)
        first_dof = self.model.body_dofadr[body_id]
        velocity = (
            np.zeros(6)
            if self.is_held(piece)
            else self.data.qvel[first_dof : first_dof + 6].copy()
        )
        return BodyPose(
            self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy(), velocity
        )

    def is_held(self, piece: FruitPiece) -> bool:
        """
        :param piece: A piece of this fruit.
        :return: Whether it is fixed in place instead of free to move.
        """
        return self.model.body_jntnum[self.body_id(piece)] == 0

    def reaches_holder(self, piece: FruitPiece, pose: BodyPose) -> bool:
        """
        :param piece: A piece of this fruit.
        :param pose: Where its body is.
        :return: Whether the piece reaches into one of :attr:`holders` there.
        """
        placement = pose.placement()
        return any(
            (part.solid.transform(placement) ^ holder).volume() > self.negligible_volume
            for part in piece.parts
            for holder in self.holders
        )

    def piece_with(self, material: FruitMaterial) -> FruitPiece:
        """
        :param material: A material of which the fruit has a single part.
        :return: The piece holding that part.
        :raises FruitPartNotFound: If no piece has a part of the material.
        """
        for piece in self.pieces:
            if material in piece.materials():
                return piece
        raise FruitPartNotFound(f"{self.body_name}_{material}")

    # %% cutting

    def materials_cut(
        self, materials: frozenset[FruitMaterial] | None
    ) -> frozenset[FruitMaterial]:
        """
        :param materials: The materials a blade is to cut; every material if None.
        :return: Those of them a knife can cut.
        """
        return frozenset(FruitMaterial if materials is None else materials) - (
            self.uncuttable
        )

    def cut(
        self,
        plane: CuttingPlane,
        pieces: list[FruitPiece] | None = None,
        materials: frozenset[FruitMaterial] | None = None,
    ) -> list[FruitPiece]:
        """
        Cut pieces of the fruit in two along a plane. The pieces of a held piece stay
        held when the cut leaves a material a knife could cut whole, as they stay stuck
        to it; for a fruit with a :attr:`texture`, they also stick to the piece they
        touch, as skin scored off the flesh does until it is peeled.

        :param plane: The plane to cut along, in the world.
        :param pieces: The pieces to cut; every piece if None.
        :param materials: The materials to cut; every one a knife can cut if None.
        :return: The pieces the cut made; none if the plane misses the fruit.
        """
        materials = self.materials_cut(materials)
        leaves_material_whole = materials != self.materials_cut(None)
        made: list[tuple[FruitPiece, BodyPose]] = []
        for piece in list(self.pieces if pieces is None else pieces):
            pose = self.pose_of(piece)
            successors = self.split(piece, plane.seen_from(pose), materials)
            if successors:
                stuck = leaves_material_whole and self.is_held(piece)
                self.replace(piece, successors, pose, stuck)
                self.stick_round_pit(pieces=successors)
                if stuck:
                    self.stick_peel(pieces=successors, materials=materials)
                made.extend((successor, pose) for successor in successors)
        if made:
            self.recompile(made)
        return [piece for piece, _ in made]

    def cut_with(
        self,
        blade: KnifeBlade,
        pieces: list[FruitPiece] | None = None,
        materials: frozenset[FruitMaterial] | None = None,
    ) -> list[FruitPiece]:
        """
        Carry on with the blade's cut into each piece, and cut along the blade's plane
        every piece whose section through the materials cut the cutting edge has, since
        the blade entered that plane, gone past everywhere.

        :param blade: The blade of a knife in :attr:`model`.
        :param pieces: The pieces the blade may cut; every piece if None.
        :param materials: The materials the blade cuts; every one a knife can cut if
            None.
        :return: The pieces the cut made.
        """
        plane = blade.plane(self.model, self.data)
        cut_through = []
        for piece in self.pieces if pieces is None else pieces:
            pose = self.pose_of(piece)
            incision = self.incision_into(piece, plane.seen_from(pose), materials)
            if incision is None:
                continue
            incision.passed |= blade.passed(
                self.model, self.data, pose.to_world(incision.samples)
            )
            if incision.passed.all():
                cut_through.append(piece)
        return self.cut(plane, cut_through, materials) if cut_through else []

    def incision_into(
        self,
        piece: FruitPiece,
        plane: CuttingPlane,
        materials: frozenset[FruitMaterial] | None = None,
    ) -> Incision | None:
        """
        :param piece: A piece of this fruit.
        :param plane: The plane of a blade, in the frame of the piece's body.
        :param materials: The materials the blade cuts; every one a knife can cut if
            None.
        :return: The blade's cut into the piece: the one begun along that plane through
            those materials, else a new one; None if the plane misses them.
        """
        materials = self.materials_cut(materials)
        incision = self.incisions.get(piece.body_name)
        if (
            incision is not None
            and incision.materials == materials
            and incision.follows(plane, self.incision_tolerance)
        ):
            return incision
        section = piece.section(plane, set(materials))
        if section is None:
            self.incisions.pop(piece.body_name, None)
            return None
        incision = Incision(
            plane,
            materials,
            self.reachable(piece, plane, section.samples(self.sample_spacing)),
        )
        self.incisions[piece.body_name] = incision
        return incision

    def reachable(
        self, piece: FruitPiece, plane: CuttingPlane, samples: np.ndarray
    ) -> np.ndarray:
        """
        :param piece: A piece of this fruit.
        :param plane: The plane of a blade, in the frame of the piece's body.
        :param samples: Points of the piece's section in that plane, one row each.
        :return: Those of the points the blade can reach: not within
            :attr:`uncuttable_margin` of a part no knife cuts, which the blade keeps
            clear of and the flesh round which comes away with the cut.
        """
        section = piece.section(plane, set(self.uncuttable))
        if section is None or not len(samples):
            return samples
        coordinates = (samples - plane.point) @ section.in_plane_axes.T
        corners = np.vstack(section.loops)
        nearest = np.linalg.norm(
            coordinates[:, None, :] - corners[None, :, :], axis=-1
        ).min(axis=1)
        return samples[
            ~(section.contains(coordinates) | (nearest < self.uncuttable_margin))
        ]

    def section_outline(
        self, plane: CuttingPlane, materials: frozenset[FruitMaterial]
    ) -> np.ndarray:
        """
        :param plane: A plane in the world.
        :param materials: The materials whose parts the section goes through.
        :return: The corners of the outline the plane cuts out of the parts of those
            materials, in the world, one row each; none if it goes through none.
        """
        outlines = [np.empty((0, 3))]
        for piece in self.pieces:
            pose = self.pose_of(piece)
            section = piece.section(plane.seen_from(pose), set(materials))
            if section is not None:
                outlines.append(pose.to_world(section.outline_points()))
        return np.vstack(outlines)

    def separate(self, piece: FruitPiece, material: FruitMaterial) -> FruitPiece:
        """
        Take the part of one material out of a piece, as a piece of its own that lies
        where it was; both stay held if the piece was.

        :param piece: A piece of this fruit.
        :param material: The material of the part to take out.
        :return: The piece holding only that part; *piece* itself if it has no other.
        """
        part = piece.part_of(material)
        if len(piece.parts) == 1:
            return piece
        pose = self.pose_of(piece)
        separated = self.new_piece([part.moved_whole()])
        remaining = self.new_piece(
            [other.moved_whole() for other in piece.parts if other is not part]
        )
        self.replace(piece, [separated, remaining], pose, self.is_held(piece))
        if material in self.uncuttable:
            self.stick(
                adhesion=self.stuck_to_pit(held=separated, holder=remaining, pit=part)
            )
        self.recompile([(separated, pose), (remaining, pose)])
        return separated

    def scoop_with(
        self,
        bowl: SpoonBowl,
        pieces: list[FruitPiece] | None = None,
        materials: frozenset[FruitMaterial] | None = None,
    ) -> list[FruitPiece]:
        """
        Follow the bowl's way through each piece, and carve out of a piece what the
        bowl held on that way once it comes out again, having been full; see
        :meth:`carve`.

        :param bowl: The bowl of a spoon in :attr:`model`.
        :param pieces: The pieces the bowl may scoop; every piece if None.
        :param materials: The materials it scoops; every one a knife can cut if None.
        :return: The scoops carved out.
        """
        materials = self.materials_cut(materials)
        bowl_pose = bowl.pose(self.model, self.data)
        scooped = []
        for piece in list(self.pieces if pieces is None else pieces):
            pose = self.pose_of(piece)
            bowl_in_piece = bowl_pose.placement_in(frame=pose)
            centre = bowl_in_piece[:, 3]
            sweep = self.sweeps.get(piece.body_name)
            if sweep is not None and (
                np.linalg.norm(centre - sweep.last_centre) < self.sweep_spacing
            ):
                continue
            lowest, highest = piece.outline.bounds
            if sweep is None and not np.all(
                np.abs(centre - (lowest + highest) / 2)
                <= (highest - lowest) / 2 + max(bowl.half_size)
            ):
                continue
            filling = self.filling(
                piece, bowl.solid().transform(bowl_in_piece), materials
            )
            space = bowl.holding_space().transform(bowl_in_piece)
            if sweep is None:
                if filling > 0:
                    self.sweeps[piece.body_name] = Sweep(space, space, centre)
                continue
            sweep.space += manifold3d.Manifold.batch_hull([sweep.last_space, space])
            sweep.last_space, sweep.last_centre = space, centre
            sweep.filled |= filling >= self.full_bowl
            if filling >= self.emerging_bowl or (filling > 0 and not sweep.filled):
                continue
            del self.sweeps[piece.body_name]
            if not sweep.filled:
                continue
            carved = self.carve(
                piece, sweep.space.transform(pose.placement()), materials
            )
            if carved is not None:
                scooped.append(carved)
                self.scoops.append(carved.body_name)
        return scooped

    def bowl_blocked(
        self,
        bowl: SpoonBowl,
        pose: BodyPose,
        materials: frozenset[FruitMaterial] | None = None,
    ) -> bool:
        """
        :param bowl: The bowl of a spoon.
        :param pose: Where the bowl's body would be.
        :param materials: The materials the spoon scoops; every one a knife can cut if
            None.
        :return: Whether the bowl would reach there into a part of any other material,
            which stops it as the pit stops a knife's edge.
        """
        materials = self.materials_cut(materials)
        solid = bowl.solid().transform(pose.placement())
        for piece in self.pieces:
            local = solid.transform(self.pose_of(piece).removal())
            lowest, highest = piece.outline.bounds
            bounds = np.array(local.bounding_box())
            if np.any(bounds[:3] > highest) or np.any(bounds[3:] < lowest):
                continue
            if any(
                (part.solid ^ local).volume() > self.negligible_volume
                for part in piece.parts
                if part.material not in materials
            ):
                return True
        return False

    def filling(
        self,
        piece: FruitPiece,
        bowl: manifold3d.Manifold,
        materials: frozenset[FruitMaterial],
    ) -> float:
        """
        :param piece: A piece of this fruit.
        :param bowl: A spoon's bowl, in the frame of the piece's body.
        :param materials: The materials the spoon scoops.
        :return: The share of the bowl's volume the piece's parts of those materials
            fill.
        """
        return (
            sum(
                (part.solid ^ bowl).volume()
                for part in piece.parts
                if part.material in materials
            )
            / bowl.volume()
        )

    def carve(
        self,
        piece: FruitPiece,
        region: manifold3d.Manifold,
        materials: frozenset[FruitMaterial] | None = None,
    ) -> FruitPiece | None:
        """
        Take what of a piece lies in a region out of it, as a free piece of its own,
        the way a spoon scoops; what is left stays where it was, held if the piece was.

        :param piece: A piece of this fruit.
        :param region: A solid in the world.
        :param materials: The materials to take; every one a knife can cut if None.
        :return: The piece taken out; None if it would be smaller than
            :attr:`smallest_piece_volume`.
        """
        materials = self.materials_cut(materials)
        pose = self.pose_of(piece)
        local = region.transform(pose.removal())
        taken = [
            share
            for share in (
                part.within(local) for part in piece.parts if part.material in materials
            )
            if share.solid.volume() > self.negligible_volume
        ]
        if sum(part.solid.volume() for part in taken) < self.smallest_piece_volume:
            return None
        left = [
            part.within(part.solid - local) if part.material in materials else part
            for part in piece.parts
        ]
        rest = [
            self.new_piece(group)
            for group in self.connected_groups(
                [part for part in left if part.solid.volume() > self.negligible_volume],
                materials,
            )
        ]
        carved = self.new_piece(taken)
        self.replace(piece, rest, pose, self.is_held(piece))
        self.add_body(carved, pose, held=False)
        self.pieces.append(carved)
        self.recompile([(carved, pose), *((remaining, pose) for remaining in rest)])
        return carved

    def hinge(self, piece: FruitPiece, planes: list[CuttingPlane]) -> list[FruitPiece]:
        """
        Cut a piece along planes into segments, each joined to the next by a hinge
        along their cut, so that it bends there as a peel does; the segments stay held
        if the piece was. A plane that does not cut the rest in two is passed over.

        :param piece: A piece of this fruit.
        :param planes: Planes in the world, in order along their common normal.
        :return: The segments, in that order.
        """
        pose = self.pose_of(piece)
        materials = self.materials_cut(None)
        segments, cuts, rest = [], [], piece
        for plane in planes:
            local = plane.seen_from(pose)
            successors = self.split(rest, local, materials)
            ahead = [
                successor
                for successor in successors
                if (successor.mass_distribution().centre - local.point) @ local.normal
                > 0
            ]
            if len(successors) != 2 or len(ahead) != 1:
                continue
            segments.extend(
                successor for successor in successors if successor not in ahead
            )
            cuts.append(local)
            rest = ahead[0]
        if not segments:
            return [piece]
        segments.append(rest)
        self.replace(piece, segments, pose, self.is_held(piece))
        for behind, ahead, cut in zip(segments, segments[1:], cuts):
            self.join_by_hinge(behind, ahead, cut)
        self.recompile([(segment, pose) for segment in segments])
        return segments

    def join_by_hinge(
        self, first: FruitPiece, second: FruitPiece, cut: CuttingPlane
    ) -> None:
        """
        Join the bodies of two pieces cut apart, both in the same pose, by a hinge
        along the longest line across their cut, and let them no longer touch.

        :param first: One piece.
        :param second: The other.
        :param cut: The plane they were cut apart along, in the frame of their bodies.
        """
        corners = PlaneSection.through(
            [part.solid for part in first.parts + second.parts], cut
        ).outline_points()
        middle = corners.mean(axis=0)
        longest = np.linalg.svd(corners - middle)[2][0]
        reach = (corners - middle) @ longest
        for end in (reach.min(), reach.max()):
            anchor = middle + end * longest
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=first.body_name,
                name2=second.body_name,
                data=np.pad([*anchor, *anchor], (0, mujoco.mjNEQDATA - 6)),
                solref=[self.hinge_time_constant, 1.0],
            )
        self.spec.add_exclude(bodyname1=first.body_name, bodyname2=second.body_name)

    def hinge_ends(self, first: FruitPiece, second: FruitPiece) -> list[np.ndarray]:
        """
        :param first: A piece.
        :param second: The piece joined to it by :meth:`hinge`.
        :return: The two ends of the line the hinge between them turns about, in the
            world, in metres, where *first* holds them.
        :raises PiecesNotHinged: If the pieces are not joined by a hinge.
        """
        body_ids = (self.body_id(first), self.body_id(second))
        model, data = self.model, self.data
        ends = [
            data.xpos[body_ids[0]]
            + data.xmat[body_ids[0]].reshape(3, 3) @ model.eq_data[joint][:3]
            for joint in range(model.neq)
            if (model.eq_obj1id[joint], model.eq_obj2id[joint]) == body_ids
        ]
        if len(ends) != 2:
            raise PiecesNotHinged(first.body_name, second.body_name)
        return ends

    def split(
        self,
        piece: FruitPiece,
        plane: CuttingPlane,
        materials: frozenset[FruitMaterial],
    ) -> list[FruitPiece]:
        """
        :param piece: A piece of this fruit.
        :param plane: A plane in the frame of the piece's body.
        :param materials: The materials to cut; the parts of others stay whole and
            together, in the piece on the side of the plane holding most of them.
        :return: The pieces the plane cuts *piece* into, each in one piece of material;
            none if it leaves the piece whole or cuts off less than
            :attr:`smallest_piece_volume`.
        """
        offset = float(plane.point @ plane.normal)
        sides: tuple[list[FruitPart], list[FruitPart]] = ([], [])
        uncut = [part for part in piece.parts if part.material not in materials]
        if uncut:
            halves = manifold3d.Manifold.compose(
                [part.solid for part in uncut]
            ).split_by_plane(plane.normal.tolist(), offset)
            larger = int(halves[1].volume() > halves[0].volume())
            sides[larger].extend(part.moved_whole() for part in uncut)
        for part in piece.parts:
            if part.material not in materials:
                continue
            halves = part.solid.split_by_plane(plane.normal.tolist(), offset)
            for side, half in zip(sides, halves):
                if half.volume() > self.negligible_volume:
                    side.append(part.within(half))
        if not (sides[0] and sides[1]):
            return []
        groups = [
            group for side in sides for group in self.connected_groups(side, materials)
        ]
        if any(
            sum(part.solid.volume() for part in group) < self.smallest_piece_volume
            for group in groups
        ):
            return []
        return [self.new_piece(group) for group in groups]

    def connected_groups(
        self, parts: list[FruitPart], materials: frozenset[FruitMaterial]
    ) -> list[list[FruitPart]]:
        """
        :param parts: Parts in one frame.
        :param materials: The materials being cut.
        :return: The parts grouped into those that hang together: a part of a material
            being cut is divided among the groups it reaches into, and the others go
            whole, together, to the group that surrounds most of them.
        """
        cut = [part for part in parts if part.material in materials]
        uncut = [part for part in parts if part.material not in materials]
        regions = FruitPart.touching_regions(cut, self.negligible_volume)
        if len(regions) <= 1:
            return [parts]
        groups = [
            [
                share
                for share in (part.within(region) for part in cut)
                if share.solid.volume() > self.negligible_volume
            ]
            for region in regions
        ]
        if not uncut:
            return groups
        enclosed = manifold3d.Manifold.compose([part.solid for part in uncut])
        surrounded = [(enclosed ^ region.hull()).volume() for region in regions]
        groups[int(np.argmax(surrounded))].extend(part.moved_whole() for part in uncut)
        return groups

    def new_piece(self, parts: list[FruitPart]) -> FruitPiece:
        """
        :param parts: The parts of a piece a cut makes.
        :return: The piece, named after the fruit and numbered.
        """
        self.pieces_made += 1
        return FruitPiece(
            f"{self.body_name}_piece_{self.pieces_made}",
            parts,
            negligible_volume=self.negligible_volume,
        )

    def materials_along(
        self, *, piece: FruitPiece, plane: CuttingPlane, points: np.ndarray
    ) -> dict[FruitMaterial, np.ndarray]:
        """
        :param piece: A piece of this fruit.
        :param plane: A plane in the world, such as a blade's.
        :param points: Points of the plane in the world, one row each, such as along a
            cutting edge.
        :return: For each material of the piece, which of the points lie in its part.
        """
        pose = self.pose_of(piece)
        local_plane = plane.seen_from(pose)
        local_points = (points - pose.position) @ pose.rotation()
        lowest, highest = piece.outline.bounds
        nearby = np.all((local_points >= lowest) & (local_points <= highest), axis=1)
        inside = {}
        for material in piece.materials():
            inside[material] = np.zeros(len(points), dtype=bool)
            if not nearby.any():
                continue
            section = self.section_in(piece=piece, material=material, plane=local_plane)
            if section is None:
                continue
            coordinates = (local_points[nearby] - section.plane.point) @ (
                section.in_plane_axes.T
            )
            inside[material][nearby] = section.contains(coordinates)
        return inside

    def section_in(
        self, *, piece: FruitPiece, material: FruitMaterial, plane: CuttingPlane
    ) -> PlaneSection | None:
        """
        :param piece: A piece of this fruit.
        :param material: One of its materials.
        :param plane: A plane in the frame of the piece's body.
        :return: The section the plane cuts out of the piece's part of that material,
            the one last worked out while the plane lies within
            :attr:`incision_tolerance` of where it lay then; None if it misses it.
        """
        key = (piece.body_name, material)
        if key in self.sections:
            known = self.sections[key]
            known_plane = known.plane if known is not None else None
            if known_plane is not None and known_plane.lies_near(
                other=plane, tolerance=self.incision_tolerance
            ):
                return known
        section = piece.section(plane, {material})
        self.sections[key] = section
        return section

    # %% sticking and bruising

    def stick_round_pit(self, *, pieces: list[FruitPiece]) -> None:
        """
        Let each of the pieces a cut made that touches the pit in another of them stick
        to that other, as halves cut round a pit stick together until twisted apart.

        :param pieces: The pieces one cut made of one piece.
        """
        for holder in pieces:
            pits = holder.parts_of(materials=self.uncuttable)
            if not pits:
                continue
            for held in pieces:
                if held is not holder and self.touches(first=held.parts, second=pits):
                    self.stick(
                        adhesion=self.stuck_to_pit(
                            held=held, holder=holder, pit=pits[0]
                        )
                    )

    def stick_peel(
        self, *, pieces: list[FruitPiece], materials: frozenset[FruitMaterial]
    ) -> None:
        """
        Let each of the pieces a cut through some materials only made that is of those
        materials only stick to one of them it touches that has other materials, as a
        strip of skin scored off the flesh sticks to it until peeled; nothing happens
        for a fruit without a :attr:`texture`.

        :param pieces: The pieces one cut made of one held piece.
        :param materials: The materials the cut went through.
        """
        if self.texture is None:
            return
        for held in pieces:
            if not held.materials() <= materials:
                continue
            holder = next(
                (
                    other
                    for other in pieces
                    if not other.materials() <= materials
                    and self.touches(first=held.parts, second=other.parts)
                ),
                None,
            )
            if holder is None:
                continue
            lowest, highest = held.outline.bounds
            width = float(np.sort(highest - lowest)[1])
            force = self.texture.peel_strength * width
            self.stick(
                adhesion=Adhesion(
                    held=held.body_name,
                    holder=holder.body_name,
                    through=materials,
                    anchor=held.mass_distribution().centre,
                    break_force=force,
                    break_torque=force * width / 2,
                    time_constant=self.hinge_time_constant,
                )
            )

    def piece_named(self, *, body_name: str) -> FruitPiece:
        """
        :param body_name: The name of the body of a piece of this fruit.
        :return: The piece.
        :raises FruitPartNotFound: If no piece has a body of that name.
        """
        for piece in self.pieces:
            if piece.body_name == body_name:
                return piece
        raise FruitPartNotFound(body_name)

    def touches(self, *, first: list[FruitPart], second: list[FruitPart]) -> bool:
        """
        :param first: Parts of pieces of this fruit.
        :param second: Other parts of them.
        :return: Whether one of the first lies within :attr:`touching_distance` of one
            of the second, as they lay in the fruit.
        """
        return any(
            mine.solid.min_gap(theirs.solid, self.touching_distance)
            < self.touching_distance
            for mine in first
            for theirs in second
        )

    def stuck_to_pit(
        self, *, held: FruitPiece, holder: FruitPiece, pit: FruitPart
    ) -> Adhesion | None:
        """
        :param held: A piece that sticks to the pit or the flesh round it.
        :param holder: The piece it sticks to.
        :param pit: The pit, in one of the two pieces.
        :return: The adhesion between them, as strong as the fruit's flesh holds its
            pit; None if the fruit has no :attr:`texture`.
        """
        if self.texture is None:
            return None
        lowest, highest = FruitPart.triangle_mesh_of(pit.solid).bounds
        radius = float(np.min(highest - lowest)) / 2
        torque = self.texture.release_torque
        return Adhesion(
            held=held.body_name,
            holder=holder.body_name,
            through=self.uncuttable,
            anchor=(lowest + highest) / 2,
            break_force=torque / radius,
            break_torque=torque,
            time_constant=self.hinge_time_constant,
        )

    def stick(self, *, adhesion: Adhesion | None) -> None:
        """
        Let two pieces stick together from the next compilation of :attr:`spec` on.

        :param adhesion: How they stick; nothing happens if None.
        """
        if adhesion is None:
            return
        adhesion.add_to(spec=self.spec)
        self.adhesions.append(adhesion)

    def carry_over(
        self, *, adhesion: Adhesion, replaced: FruitPiece, successors: list[FruitPiece]
    ) -> None:
        """
        Let each piece cut from a piece that stuck to another stick to that other as
        before, where a part of what held them together touches the other.

        :param adhesion: How the cut piece stuck.
        :param replaced: The cut piece.
        :param successors: The pieces it was cut into.
        """
        held = adhesion.held == replaced.body_name
        other = self.piece_named(body_name=adhesion.holder if held else adhesion.held)
        for successor in successors:
            if not (
                self.touches(
                    first=successor.parts_of(materials=adhesion.through),
                    second=other.parts,
                )
                or self.touches(
                    first=successor.parts,
                    second=other.parts_of(materials=adhesion.through),
                )
            ):
                continue
            names = (successor.body_name, other.body_name)
            self.stick(
                adhesion=adhesion.joining(held=names[not held], holder=names[held])
            )

    def break_adhesions(self) -> list[Adhesion]:
        """
        Let go of every adhesion that carries more than it can or has been stretched
        apart, after a step, and let the pieces it held touch again.

        .. warning:: When an adhesion between pieces of which one moves under forces
            breaks, :attr:`model` and :attr:`data` are replaced, as after a cut.

        :return: The adhesions that broke.
        """
        broken = [
            adhesion
            for adhesion in self.adhesions
            if adhesion.is_breaking(model=self.model, data=self.data)
        ]
        for adhesion in broken:
            weld_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, adhesion.name
            )
            self.data.eq_active[weld_id] = False
            adhesion.remove_from(spec=self.spec)
            self.adhesions.remove(adhesion)
        # Only pieces that move under forces have contacts to give back.
        if any(
            not self.is_held(self.piece_named(body_name=name))
            for adhesion in broken
            for name in (adhesion.held, adhesion.holder)
        ):
            self.recompile([])
        return broken

    def contact_patches(self) -> dict[tuple[int, int], ContactPatch]:
        """
        :return: Each patch where two geoms press together after a step, by the ids of
            the two geoms, the smaller first.
        """
        patches = {}
        force = np.zeros(6)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            mujoco.mj_contactForce(self.model, self.data, contact_id, force)
            if force[0] <= 0:
                continue
            first, second = sorted(int(geom_id) for geom_id in contact.geom)
            patch = patches.setdefault((first, second), ContactPatch())
            patch.add_contact(force=float(force[0]), position=contact.pos)
        return patches

    def record_bruises(self) -> list[Bruise]:
        """
        Record where a contact patch with something other than this fruit's pieces,
        whose convex shapes may overlap where they bridge a cavity, presses a piece
        harder than its flesh bears, after a step; a bruise next to one already recorded
        deepens that one.

        :return: The bruises recorded or deepened.
        """
        if self.texture is None:
            return []
        pieces = {self.body_id(piece): piece for piece in self.pieces}
        recorded = []
        for geoms, patch in self.contact_patches().items():
            touched = [pieces.get(self.model.geom_bodyid[geom_id]) for geom_id in geoms]
            if None not in touched:
                continue
            pressure = self.texture.contact_pressure(
                force=patch.force, radius=self.contact_radius
            )
            if pressure <= self.texture.bruising_pressure:
                continue
            for piece in touched:
                if piece is None or FruitMaterial.FLESH not in piece.materials():
                    continue
                point = self.pose_of(piece).removal() @ np.append(patch.centre, 1.0)
                recorded.append(self.deepen_bruise(point=point, pressure=pressure))
        return recorded

    def deepen_bruise(self, *, point: np.ndarray, pressure: float) -> Bruise:
        """
        :param point: Where a press bruises the fruit, in its frame.
        :param pressure: How hard, in pascals.
        :return: The bruise already recorded whose reach the point lies in, its
            pressure raised to this one's if higher; else a new bruise.
        """
        for bruise in self.bruises:
            if np.linalg.norm(bruise.point - point) <= bruise.reach:
                bruise.pressure = max(bruise.pressure, pressure)
                return bruise
        bruise = Bruise(point=point, pressure=pressure)
        self.bruises.append(bruise)
        return bruise

    # %% rebuilding the model

    def replace(
        self,
        piece: FruitPiece,
        successors: list[FruitPiece],
        pose: BodyPose,
        stuck: bool,
    ) -> None:
        """
        Put the bodies of new pieces into :attr:`spec` in place of a piece's body.

        :param piece: The piece to take out.
        :param successors: The pieces to put in.
        :param pose: Where their bodies go.
        :param stuck: Whether they stay held even where they reach into no holder.
        """
        self.incisions.pop(piece.body_name, None)
        for material in piece.materials():
            self.sections.pop((piece.body_name, material), None)
        carried = [
            adhesion
            for adhesion in self.adhesions
            if piece.body_name in (adhesion.held, adhesion.holder)
        ]
        for adhesion in carried:
            adhesion.remove_from(spec=self.spec)
            self.adhesions.remove(adhesion)
        self.spec.delete(self.spec.body(piece.body_name))
        for mesh_name in piece.mesh_names:
            self.spec.delete(self.spec.mesh(mesh_name))
        for successor in successors:
            self.add_body(
                successor, pose, stuck or self.reaches_holder(successor, pose)
            )
        index = self.pieces.index(piece)
        self.pieces[index : index + 1] = successors
        for adhesion in carried:
            self.carry_over(adhesion=adhesion, replaced=piece, successors=successors)

    def add_body(self, piece: FruitPiece, pose: BodyPose, held: bool) -> None:
        """
        Add a body for a piece to :attr:`spec`: a geom for each part and the convex
        shapes the piece collides through, which give under a press as the
        :meth:`contact_time_constant` of the piece makes them.

        :param piece: The piece.
        :param pose: Where the body goes.
        :param held: Whether the body is fixed there instead of free.
        """
        distribution = piece.mass_distribution()
        inertia = distribution.inertia
        body = self.spec.worldbody.add_body(
            name=piece.body_name,
            pos=pose.position,
            quat=pose.quaternion,
            mass=distribution.mass,
            ipos=distribution.centre,
            fullinertia=[
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
            ],
            explicitinertial=True,
        )
        if not held:
            body.add_freejoint(name=f"{piece.body_name}_free")
        shapes = [
            (f"{piece.body_name}_{part.material}", part.geom_class, mesh)
            for part in piece.parts
            for mesh in [FruitPart.triangle_mesh_of(part.solid)]
        ]
        shapes.extend(
            (f"{piece.body_name}_collision_{number}", self.collision_class, hull)
            for number, hull in enumerate(self.collision_shapes(piece))
        )
        for name, geom_class, mesh in shapes:
            self.spec.add_mesh(
                name=name,
                uservert=np.asarray(mesh.vertices, dtype=float).ravel(),
                userface=np.asarray(mesh.faces, dtype=int).ravel(),
            )
            piece.mesh_names.append(name)
            body.add_geom(
                self.spec.find_default(geom_class),
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
            )
        if self.texture is None:
            return
        time_constant = self.contact_time_constant(piece=piece)
        for geom in body.geoms:
            if geom.classname.name == self.collision_class:
                geom.solref = [time_constant, 1.0]

    def collision_shapes(self, piece: FruitPiece) -> list[trimesh.Trimesh]:
        """
        :param piece: A piece of this fruit.
        :return: Convex shapes that together fill the parts of the piece that do not
            collide as they are, in the frame of the piece's body, and that stay out of
            the material of other pieces.
        """
        shaped = [part for part in piece.parts if not part.collides_as_is]
        if not shaped:
            return []
        solid = manifold3d.Manifold.compose(
            FruitPart.touching_regions(shaped, self.negligible_volume)
        )
        others = manifold3d.Manifold.compose(
            [part.solid for part in piece.parts if part.collides_as_is]
        )
        foreign = self.material - solid - others
        return [
            FruitPart.triangle_mesh_of(shape)
            for shape in self.convex_cover(solid, foreign, None, self.refinements)
        ]

    def convex_cover(
        self,
        solid: manifold3d.Manifold,
        foreign: manifold3d.Manifold,
        directions: np.ndarray | None,
        refinements: int,
    ) -> list[manifold3d.Manifold]:
        """
        :param solid: A solid in the fruit's frame.
        :param foreign: The material the cover has to stay out of.
        :param directions: The three unit vectors from :attr:`core` that bound the
            region holding *solid*, one row each; None if it is not bounded.
        :param refinements: How many times the region may still be divided.
        :return: Convex shapes that cover *solid*: its convex hull where that stays out
            of *foreign*, else the covers of what lies in each part of the region.
        """
        hull = solid.hull()
        if not refinements or (hull ^ foreign).volume() <= self.negligible_volume:
            return [hull]
        if directions is None:
            icosahedron = trimesh.creation.icosahedron()
            regions = icosahedron.vertices[icosahedron.faces]
            regions /= np.linalg.norm(regions, axis=-1, keepdims=True)
        else:
            middles = directions + np.roll(directions, -1, axis=0)
            middles /= np.linalg.norm(middles, axis=-1, keepdims=True)
            regions = np.array(
                [
                    [directions[0], middles[0], middles[2]],
                    [directions[1], middles[1], middles[0]],
                    [directions[2], middles[2], middles[1]],
                    middles,
                ]
            )
        cover = []
        for region in regions:
            cone = manifold3d.Manifold.hull_points(
                np.vstack([self.core, self.core + self.reach * region])
            )
            inside = solid ^ cone
            if inside.volume() > self.negligible_volume:
                cover.extend(
                    self.convex_cover(inside, foreign ^ cone, region, refinements - 1)
                )
        return cover

    def release(self, piece: FruitPiece) -> None:
        """
        Let a held or gripped piece move freely from where it is.

        :param piece: A piece of this fruit.
        """
        if not self.is_held(piece):
            return
        pose = self.pose_of(piece)
        body = self.spec.body(piece.body_name)
        body.mocap = False
        body.add_freejoint(name=f"{piece.body_name}_free")
        self.recompile([(piece, pose)])

    def hold(self, piece: FruitPiece) -> None:
        """
        Fix a free piece in place where it is, as when it is pushed onto a holder.

        :param piece: A piece of this fruit.
        """
        if self.is_held(piece):
            return
        pose = self.pose_of(piece)
        body = self.spec.body(piece.body_name)
        body.pos = pose.position
        body.quat = pose.quaternion
        for joint in body.joints:
            self.spec.delete(joint)
        self.recompile([])

    def grip(self, piece: FruitPiece) -> None:
        """
        Hold a piece where it is, as a hand does: from then on it moves only where its
        mocap body, named after the piece, is put, until it is released.

        :param piece: A piece of this fruit.
        """
        self.hold(piece)
        self.spec.body(piece.body_name).mocap = True
        self.recompile([])

    def recompile(self, placed: list[tuple[FruitPiece, BodyPose]]) -> None:
        """
        Compile :attr:`spec` into a new :attr:`model` and :attr:`data` that carry on
        from the old state.

        :param placed: The pieces added or released since the last compilation, each
            with where its body is and how it moves.
        """
        self.model, self.data = self.spec.recompile(self.model, self.data)
        self.data.eq_active[:] = self.model.eq_active0
        for piece, pose in placed:
            if self.is_held(piece):
                continue
            body_id = self.body_id(piece)
            joint_id = self.model.body_jntadr[body_id]
            first_position = self.model.jnt_qposadr[joint_id]
            first_dof = self.model.jnt_dofadr[joint_id]
            self.data.qpos[first_position : first_position + 3] = pose.position
            self.data.qpos[first_position + 3 : first_position + 7] = pose.quaternion
            self.data.qvel[first_dof : first_dof + 6] = pose.velocity
        mujoco.mj_forward(self.model, self.data)
