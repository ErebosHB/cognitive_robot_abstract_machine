"""
The PR2 in a MuJoCo scene of the avocado demonstrations: exported from the PR2 of the
semantic digital twin, and arms moved by inverse kinematics onto their position servos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import mujoco
import numpy as np
from semantic_digital_twin.adapters.multi_sim import MujocoBuilder
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification

# %% names


class ArmSide(StrEnum):
    """
    The PR2's arms, by the prefix of their names.
    """

    LEFT = "l"
    RIGHT = "r"


class ArmJoint(StrEnum):
    """
    The joints of a PR2 arm from shoulder to wrist, without the arm's prefix.
    """

    SHOULDER_PAN = "shoulder_pan_joint"
    SHOULDER_LIFT = "shoulder_lift_joint"
    UPPER_ARM_ROLL = "upper_arm_roll_joint"
    ELBOW_FLEX = "elbow_flex_joint"
    FOREARM_ROLL = "forearm_roll_joint"
    WRIST_FLEX = "wrist_flex_joint"
    WRIST_ROLL = "wrist_roll_joint"


class Pr2Name(StrEnum):
    """
    Names of the exported PR2's parts, without the prefix it is attached with.
    """

    TORSO = "torso_lift_joint"
    HEAD_PAN = "head_pan_joint"
    HEAD_TILT = "head_tilt_joint"
    SERVO_SUFFIX = "_servo"
    TOOL_FRAME = "_gripper_tool_frame"
    FINGER = "_gripper_l_finger_joint"
    DRIVE_PREFIX = "pr2_omni_drive_"


# %% the model


@dataclass
class Pr2Export:
    """
    Writes the PR2 of the semantic digital twin as a MuJoCo model, with the same meshes,
    joints and servos the simulations of CRAM give it.
    """

    file_name: str = "pr2.xml"
    """
    The name of the written model file.
    """

    def write(self, directory: Path) -> Path:
        """
        :param directory: An empty folder to write the model and its meshes into.
        :return: The model file.
        """
        world = WorldSpecification(objects=[]).create_world()
        RobotSpecification(PR2).spawn(world)
        path = directory.resolve() / self.file_name
        MujocoBuilder().build_world(world=world, file_path=str(path))
        return path

    def standing_spec(self, path: Path) -> mujoco.MjSpec:
        """
        Load the written model ready to be attached to a scene: its base fixed where it
        is attached, its weight carried by gravity compensation, as the PR2's
        counterbalanced arms are, and without keyframes, which would not fit the scene.

        :param path: The model file :meth:`write` wrote.
        :return: The model to attach.
        """
        spec = mujoco.MjSpec.from_file(str(path))
        for actuator in list(spec.actuators):
            if actuator.target.startswith(Pr2Name.DRIVE_PREFIX):
                spec.delete(actuator)
        for joint in list(spec.joints):
            if joint.name.startswith(Pr2Name.DRIVE_PREFIX):
                spec.delete(joint)
        for key in list(spec.keys):
            spec.delete(key)
        for body in spec.bodies:
            body.gravcomp = 1.0
        return spec


# %% the arms


@dataclass
class Pr2Arm:
    """
    One arm of a PR2 in a compiled scene, moved by setting the targets of its servos.
    """

    model: mujoco.MjModel
    """
    The compiled scene.
    """

    data: mujoco.MjData
    """
    The simulation state of :attr:`model`.
    """

    side: ArmSide
    """
    Which arm this is.
    """

    prefix: str
    """
    The prefix the PR2's names were given in the scene.
    """

    damping: float = 0.03
    """
    The damping of the least-squares inverse kinematics, keeping it calm near
    singularities.
    """

    posture_gain: float = 0.1
    """
    How strongly each step of the inverse kinematics pulls the arm toward
    :attr:`posture`, within the freedom the target leaves.
    """

    elbow_out_margin: float = 0.2
    """
    How far in radians from zero the upper arm's roll stays, on the arm's own side, so
    the elbow points outward and away from the other arm.
    """

    iterations: int = 25
    """
    The most steps the inverse kinematics takes; it starts from the last solution, and
    the target moves little between two solutions.
    """

    tolerance: float = 5e-4
    """
    The pose error, in metres and radians, at which the inverse kinematics stops.
    """

    joint_ids: np.ndarray = field(init=False)
    """
    The ids of the arm's joints, from shoulder to wrist.
    """

    actuator_ids: np.ndarray = field(init=False)
    """
    The ids of the servos of :attr:`joint_ids`.
    """

    tool_id: int = field(init=False)
    """
    The id of the body between the fingertips.
    """

    finger_actuator_id: int = field(init=False)
    """
    The id of the servo opening and closing the gripper.
    """

    hand_geom_ids: np.ndarray = field(init=False)
    """
    The ids of the gripper's geoms, palm and fingers.
    """

    hand_collision_bits: np.ndarray = field(init=False)
    """
    The contype and conaffinity of :attr:`hand_geom_ids`, one row each, as compiled.
    """

    posture: np.ndarray = field(init=False)
    """
    The joint positions the arm leans toward, those it starts in.
    """

    solution: np.ndarray = field(init=False)
    """
    The joint positions last solved for.
    """

    kinematics: mujoco.MjData = field(init=False)
    """
    Scratch state in which the inverse kinematics moves the arm.
    """

    def __post_init__(self) -> None:
        names = [f"{self.prefix}{self.side}_{joint}" for joint in ArmJoint]
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        )
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + Pr2Name.SERVO_SUFFIX
                )
                for n in names
            ]
        )
        self.tool_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{self.prefix}{self.side}{Pr2Name.TOOL_FRAME}",
        )
        self.finger_actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{self.prefix}{self.side}{Pr2Name.FINGER}{Pr2Name.SERVO_SUFFIX}",
        )
        hand_prefix = f"{self.prefix}{self.side}_gripper"
        self.hand_geom_ids = np.array(
            [
                geom
                for geom in range(self.model.ngeom)
                if self.model.body(self.model.geom_bodyid[geom]).name.startswith(
                    hand_prefix
                )
            ]
        )
        self.hand_collision_bits = np.column_stack(
            [
                self.model.geom_contype[self.hand_geom_ids],
                self.model.geom_conaffinity[self.hand_geom_ids],
            ]
        )
        self.kinematics = mujoco.MjData(self.model)
        self.posture = self.joint_positions()
        self.solution = self.posture.copy()

    @property
    def qpos_addresses(self) -> np.ndarray:
        """
        The positions of the arm's joints in ``qpos``.
        """
        return self.model.jnt_qposadr[self.joint_ids]

    @property
    def dof_addresses(self) -> np.ndarray:
        """
        The positions of the arm's joints in ``qvel``.
        """
        return self.model.jnt_dofadr[self.joint_ids]

    def joint_positions(self) -> np.ndarray:
        """
        :return: Where the arm's joints are now.
        """
        return self.data.qpos[self.qpos_addresses].copy()

    def place_joints(self, positions: np.ndarray) -> None:
        """
        Put the arm at *positions* and make them its servos' targets and its posture;
        for setting up the scene before the simulation starts.

        :param positions: The joint positions from shoulder to wrist.
        """
        self.data.qpos[self.qpos_addresses] = positions
        self.data.ctrl[self.actuator_ids] = positions
        self.posture = np.array(positions, dtype=float)
        self.solution = self.posture.copy()

    def tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        :return: The position and rotation matrix of the point between the fingertips.
        """
        return (
            self.data.xpos[self.tool_id].copy(),
            self.data.xmat[self.tool_id].reshape(3, 3).copy(),
        )

    def solve(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """
        Find joint positions putting the tool frame at a pose, starting from the last
        solution, by damped least squares within the joints' limits.

        :param position: The wanted tool position, in world coordinates.
        :param rotation: The wanted tool rotation matrix, in world axes.
        :return: The joint positions; also kept as :attr:`solution`.
        """
        state = self.kinematics
        state.qpos[:] = self.data.qpos
        state.qpos[self.qpos_addresses] = self.solution
        limited = self.model.jnt_limited[self.joint_ids].astype(bool)
        lower, upper = self.model.jnt_range[self.joint_ids].T.copy()
        roll = list(ArmJoint).index(ArmJoint.UPPER_ARM_ROLL)
        if self.side == ArmSide.LEFT:
            lower[roll] = max(lower[roll], self.elbow_out_margin)
        else:
            upper[roll] = min(upper[roll], -self.elbow_out_margin)
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        for _ in range(self.iterations):
            mujoco.mj_kinematics(self.model, state)
            mujoco.mj_comPos(self.model, state)
            current = state.xmat[self.tool_id].reshape(3, 3)
            error = np.concatenate(
                [
                    position - state.xpos[self.tool_id],
                    0.5 * np.cross(current.T, rotation.T).sum(axis=0),
                ]
            )
            if np.linalg.norm(error) < self.tolerance:
                break
            mujoco.mj_jacBody(
                self.model, state, jacobian_position, jacobian_rotation, self.tool_id
            )
            jacobian = np.vstack([jacobian_position, jacobian_rotation])[
                :, self.dof_addresses
            ]
            inverse = jacobian.T @ np.linalg.inv(
                jacobian @ jacobian.T + self.damping**2 * np.eye(6)
            )
            arm = state.qpos[self.qpos_addresses]
            step = inverse @ error + (np.eye(len(arm)) - inverse @ jacobian) @ (
                self.posture_gain * (self.posture - arm)
            )
            arm = arm + np.clip(step, -0.2, 0.2)
            arm[limited] = np.clip(arm[limited], lower[limited], upper[limited])
            state.qpos[self.qpos_addresses] = arm
        self.solution = state.qpos[self.qpos_addresses].copy()
        return self.solution

    def command(self, positions: np.ndarray) -> None:
        """
        :param positions: The joint positions the servos are to hold.
        """
        self.data.ctrl[self.actuator_ids] = positions

    def set_gripper(self, opening: float) -> None:
        """
        :param opening: The finger angle in radians, 0 closed; clipped to the servo's range.
        """
        low, high = self.model.actuator_ctrlrange[self.finger_actuator_id]
        self.data.ctrl[self.finger_actuator_id] = np.clip(opening, low, high)

    def gripper_opening(self) -> float:
        """
        :return: The finger angle in radians the gripper is at now.
        """
        joint = self.model.actuator_trnid[self.finger_actuator_id][0]
        return float(self.data.qpos[self.model.jnt_qposadr[joint]])

    def set_hand_collisions(self, contype: int | None, conaffinity: int = 0) -> None:
        """
        :param contype: The contact type the gripper's geoms take, or None for those
            they were compiled with (and then their compiled affinity too).
        :param conaffinity: The contact affinity they take with a given *contype*.
        """
        if contype is None:
            bits = self.hand_collision_bits
        else:
            bits = np.tile([contype, conaffinity], (len(self.hand_geom_ids), 1))
        self.model.geom_contype[self.hand_geom_ids] = bits[:, 0]
        self.model.geom_conaffinity[self.hand_geom_ids] = bits[:, 1]
