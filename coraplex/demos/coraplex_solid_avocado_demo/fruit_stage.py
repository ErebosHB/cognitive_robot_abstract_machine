"""
The pieces of a fruit's world as a USD stage that Isaac Sim can open and simulate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from semantic_digital_twin.world import Body

from fruit_world import FruitWorld

# %% the stage


@dataclass
class FruitStage:
    """
    A USD stage, in metres with z up, holding each piece of a :class:`FruitWorld` as a
    rigid body where it is now, weighing what it weighs. Its parts are meshes that are
    only seen, and the convex shapes it collides through are meshes only collided with.
    """

    fruit_world: FruitWorld
    """
    The world of the pieces.
    """

    floor_height: float | None = None
    """
    The height in metres of a static floor the pieces lie on; None for no floor.
    """

    floor_size: float = 2.0
    """
    The length of the floor's sides, in metres.
    """

    gravity: float = 9.81
    """
    The acceleration of gravity, pointing down, in metres per second squared.
    """

    def write(self, path: Path) -> None:
        """
        Write the stage, of the pieces as they are now, to a file.

        :param path: The file; its extension, such as ``.usda`` or ``.usd``, chooses
            the format.
        """
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
        scene = UsdPhysics.Scene.Define(stage, "/World/physics")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(self.gravity)
        if self.floor_height is not None:
            self.add_floor(stage)
        for connection in self.fruit_world.connections.values():
            self.add_piece(stage, connection.child)
        stage.Export(str(path))

    def add_floor(self, stage: Usd.Stage) -> None:
        """
        Add a static floor at :attr:`floor_height` to a stage.

        :param stage: The stage.
        """
        floor = UsdGeom.Plane.Define(stage, "/World/floor")
        floor.CreateAxisAttr(UsdGeom.Tokens.z)
        floor.CreateWidthAttr(self.floor_size)
        floor.CreateLengthAttr(self.floor_size)
        floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, self.floor_height))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    def add_piece(self, stage: Usd.Stage, body: Body) -> None:
        """
        Add a piece's rigid body, its parts and its collision shapes to a stage.

        :param stage: The stage.
        :param body: The piece's body in :attr:`fruit_world`.
        """
        world = self.fruit_world.world
        path = f"/World/{body.name.name}"
        piece = UsdGeom.Xform.Define(stage, path)
        transform = world.compute_forward_kinematics_np(world.root, body)
        piece.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*transform[:3, 3])
        )
        piece.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(*self.quaternion_of(transform[:3, :3]))
        )
        UsdPhysics.RigidBodyAPI.Apply(piece.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(piece.GetPrim())
        inertial = body.inertial
        moments, axes = inertial.inertia.to_principal_moments_and_axes()
        mass.CreateMassAttr(inertial.mass)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*inertial.center_of_mass.to_np()[:3]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments.to_values()))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(*self.quaternion_of(axes.data)))
        for part in body.child_kinematic_structure_entities:
            for shape in part.visual.shapes:
                mesh = self.add_mesh(
                    stage, f"{path}/{part.name.name}", shape.mesh_in_frame(body)
                )
                mesh.CreateDisplayColorAttr([Gf.Vec3f(*shape.color.to_rgb())])
        for number, shape in enumerate(body.collision.shapes):
            mesh = self.add_mesh(
                stage, f"{path}/collision_{number}", shape.mesh_in_frame(body)
            )
            mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
                UsdPhysics.Tokens.convexHull
            )

    @staticmethod
    def add_mesh(stage: Usd.Stage, path: str, surface: trimesh.Trimesh) -> UsdGeom.Mesh:
        """
        :param stage: The stage.
        :param path: Where on the stage the mesh goes.
        :param surface: The triangles of the mesh, in the frame of the prim it goes
            under.
        :return: The mesh, added to the stage.
        """
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*vertex) for vertex in surface.vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(surface.faces))
        mesh.CreateFaceVertexIndicesAttr(surface.faces.ravel().tolist())
        mesh.CreateExtentAttr([Gf.Vec3f(*corner) for corner in surface.bounds])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        return mesh

    @staticmethod
    def quaternion_of(rotation: np.ndarray) -> np.ndarray:
        """
        :param rotation: A rotation matrix.
        :return: The rotation as a unit quaternion, scalar first.
        """
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).ravel())
        return quaternion
