"""
A pit of random size, place and orientation for the solid avocado, with the cavity in
its flesh shaped to fit it, so that every run cuts round another pit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import manifold3d
import mujoco
import numpy as np
import trimesh

from cutting import FruitMaterial, FruitPart
from exceptions import PitDoesNotFit

# %% the random pit


@dataclass
class RandomPit:
    """
    How far a random pit may differ from the solid avocado's own. The fruit's frame has
    its origin at the base and its z axis along the long axis to the stem.
    """

    size_range: tuple[float, float] = (0.85, 1.1)
    """
    The smallest and the largest scale of the pit against the avocado's own.
    """

    axial_shift: float = 0.004
    """
    How far in metres the pit may move along the long axis either way.
    """

    radial_shift: float = 0.002
    """
    How far in metres the pit may move across the long axis either way, along each of
    the two axes across it.
    """

    tilt: float = 0.35
    """
    How far in radians the pit may turn, about an axis of random direction.
    """

    clearance: float = 0.0006
    """
    How much room in metres the flesh leaves round the pit, as in the avocado's own
    model, so that the convex shapes the flesh collides through stay clear of the pit.
    """

    flesh_margin: float = 0.001
    """
    How thick in metres the flesh between the pit's cavity and the skin has to stay at
    least.
    """

    negligible_volume: float = 1e-10
    """
    The volume in cubic metres of a pit's room outside the flesh below which it is a
    numerical leftover and the pit fits.
    """

    attempts: int = 20
    """
    How many random pits are tried before giving up on finding one that fits.
    """

    def apply_to(self, spec: mujoco.MjSpec, body_name: str, seed: int) -> None:
        """
        Replace the meshes of the pit and the flesh of the fruit's body in a scene by a
        random pit and flesh with a cavity that fits it.

        :param spec: The scene, not yet compiled, its meshes read from their files.
        :param body_name: The name of the fruit's body, whose geoms are named after it
            and a :class:`FruitMaterial`.
        :param seed: The seed of the random choice; the same seed gives the same pit.
        :raises PitDoesNotFit: If no pit tried fits into the flesh with its margin.
        """
        pit_mesh = self.mesh_of(spec, body_name, FruitMaterial.PIT)
        flesh_mesh = self.mesh_of(spec, body_name, FruitMaterial.FLESH)
        own_pit = self.triangle_mesh_from_file(spec, pit_mesh)
        pit = FruitPart.solid_of(own_pit)
        radius = float(np.min(own_pit.extents)) / 2
        centre = np.array(own_pit.center_mass)
        # The flesh with its cavity filled in, generously, so no sliver of it is left.
        whole_flesh = FruitPart.solid_of(
            self.triangle_mesh_from_file(spec, flesh_mesh)
        ) + self.grown(pit, centre, radius, 2 * self.clearance)
        generator = np.random.default_rng(seed)
        for _ in range(self.attempts):
            placed, placed_centre = self.random_placement(pit, centre, generator)
            placed_radius = radius * float(np.linalg.norm(placed[:, 0]))
            pit_placed = pit.transform(placed)
            room = self.grown(
                pit_placed,
                placed_centre,
                placed_radius,
                self.clearance + self.flesh_margin,
            )
            if (room - whole_flesh).volume() > self.negligible_volume:
                continue
            cavity = self.grown(
                pit_placed, placed_centre, placed_radius, self.clearance
            )
            self.replace_mesh(spec, pit_mesh, pit_placed)
            self.replace_mesh(spec, flesh_mesh, whole_flesh - cavity)
            return
        raise PitDoesNotFit(seed, self.attempts)

    def random_placement(
        self,
        pit: manifold3d.Manifold,
        centre: np.ndarray,
        generator: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        :param pit: The avocado's own pit, in the fruit's frame.
        :param centre: Its centre of mass, in the fruit's frame, in metres.
        :param generator: The source of randomness.
        :return: The 3x4 matrix that scales, turns and moves the pit about its centre
            into a random pose, and where its centre then is.
        """
        scale = generator.uniform(*self.size_range)
        axis = generator.normal(size=3)
        angle = generator.uniform(0.0, self.tilt)
        turning = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]
        shift = np.array(
            [
                generator.uniform(-self.radial_shift, self.radial_shift),
                generator.uniform(-self.radial_shift, self.radial_shift),
                generator.uniform(-self.axial_shift, self.axial_shift),
            ]
        )
        linear = scale * turning
        moved_centre = centre + shift
        return np.column_stack([linear, moved_centre - linear @ centre]), moved_centre

    @staticmethod
    def grown(
        solid: manifold3d.Manifold, centre: np.ndarray, radius: float, distance: float
    ) -> manifold3d.Manifold:
        """
        :param solid: A solid, about as round as a pit.
        :param centre: Its middle, in metres.
        :param radius: Its smallest radius, in metres.
        :param distance: How far its narrowest part is to grow, in metres.
        :return: The solid scaled up about its middle by that much.
        """
        return (
            solid.translate((-centre).tolist())
            .scale([1 + distance / radius] * 3)
            .translate(centre.tolist())
        )

    @staticmethod
    def mesh_of(
        spec: mujoco.MjSpec, body_name: str, material: FruitMaterial
    ) -> mujoco.MjsMesh:
        """
        :param spec: The scene.
        :param body_name: The name of the fruit's body.
        :param material: A material of the fruit.
        :return: The mesh of the fruit's part of that material.
        """
        return spec.mesh(spec.geom(f"{body_name}_{material}").meshname)

    @staticmethod
    def triangle_mesh_from_file(
        spec: mujoco.MjSpec, mesh: mujoco.MjsMesh
    ) -> trimesh.Trimesh:
        """
        :param spec: The scene, read from a file.
        :param mesh: A mesh of the scene read from a file, whose geoms place it as it
            is.
        :return: The mesh as the file holds it.
        """
        path = Path(spec.modelfiledir) / spec.meshdir / mesh.file
        return trimesh.load(path, process=False, force="mesh")

    @staticmethod
    def replace_mesh(
        spec: mujoco.MjSpec, mesh: mujoco.MjsMesh, solid: manifold3d.Manifold
    ) -> None:
        """
        :param spec: The scene.
        :param mesh: A mesh of the scene.
        :param solid: The solid whose surface the mesh is to have instead.
        """
        surface = FruitPart.triangle_mesh_of(solid)
        mesh.file = ""
        mesh.uservert = np.asarray(surface.vertices, dtype=float).ravel()
        mesh.userface = np.asarray(surface.faces, dtype=int).ravel()
