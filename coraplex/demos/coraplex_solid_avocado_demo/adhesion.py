"""
Pieces of a fruit that stick together, as a pit sticks in the flesh round it or skin
sticks to flesh, until they are twisted, pulled or moved apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from fruit_material import FruitMaterial

# %% one adhesion


@dataclass(kw_only=True)
class Adhesion:
    """
    Two pieces stuck together by a weld that breaks once it carries more than a force or
    a torque, or is stretched apart.

    Both pieces' bodies have the frame of the fruit they were cut from, so the weld
    holds those frames together.
    """

    held: str
    """
    The name of the body of the piece that sticks, such as the pit.
    """

    holder: str
    """
    The name of the body of the piece it sticks to, such as the flesh round it.
    """

    through: frozenset[FruitMaterial]
    """
    The materials whose parts hold the two pieces together where they touch the other
    piece, such as the pit, which holds both halves of a fruit cut round it.
    """

    anchor: np.ndarray
    """
    The point in the frame of both bodies the weld holds them at, in metres.
    """

    break_force: float
    """
    The force in newtons above which the weld breaks.
    """

    break_torque: float
    """
    The torque in newton metres above which the weld breaks.
    """

    break_distance: float = 0.002
    """
    How far in metres the frames of the two bodies may drift apart before the weld
    tears, as it does when one of them is moved by a hand that no force stops.
    """

    break_angle: float = 0.05
    """
    How far in radians the frames of the two bodies may turn apart before the weld
    tears.
    """

    time_constant: float = 0.004
    """
    How fast in seconds the weld pulls the two bodies back together.
    """

    name: str = field(init=False)
    """
    The name of the weld in the MuJoCo scene.
    """

    def __post_init__(self) -> None:
        """
        Name the weld after the two bodies.
        """
        self.name = f"adhesion_{self.held}_{self.holder}"

    def joining(self, *, held: str, holder: str) -> Adhesion:
        """
        :param held: The name of the body that is to stick instead of :attr:`held`.
        :param holder: The name of the body it is to stick to instead of
            :attr:`holder`.
        :return: An adhesion as strong as this one between the two bodies.
        """
        return Adhesion(
            held=held,
            holder=holder,
            through=self.through,
            anchor=self.anchor,
            break_force=self.break_force,
            break_torque=self.break_torque,
            break_distance=self.break_distance,
            break_angle=self.break_angle,
            time_constant=self.time_constant,
        )

    def add_to(self, *, spec: mujoco.MjSpec) -> None:
        """
        Weld the two bodies together and let them no longer touch.

        :param spec: The scene holding both bodies, in which they lie as stuck.
        """
        # The weld holds the anchor on each body, in that body's frame, together.
        weld = [*self.anchor, *self.anchor, 1.0, 0.0, 0.0, 0.0, 1.0]
        spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name=self.name,
            name1=self.held,
            name2=self.holder,
            data=np.pad(weld, (0, mujoco.mjNEQDATA - len(weld))),
            solref=[self.time_constant, 1.0],
        )
        spec.add_exclude(name=self.name, bodyname1=self.held, bodyname2=self.holder)

    def remove_from(self, *, spec: mujoco.MjSpec) -> None:
        """
        Take the weld out and let the two bodies touch again.

        :param spec: The scene the weld was added to.
        """
        spec.delete(spec.equality(self.name))
        spec.delete(spec.exclude(self.name))

    def load(self, *, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """
        :param model: The compiled scene holding the weld.
        :param data: Its simulation state after a step.
        :return: The force in newtons and the torque in newton metres the weld carries,
            the first three and the last three entries.
        """
        weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, self.name)
        rows = (data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (
            data.efc_id[: data.nefc] == weld_id
        )
        load = np.zeros(6)
        forces = data.efc_force[: data.nefc][rows]
        load[: len(forces)] = forces
        return load

    def drift(
        self, *, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[float, float]:
        """
        :param model: The compiled scene holding both bodies.
        :param data: Its simulation state.
        :return: How far in metres and how far in radians the frames of the two bodies
            have drifted apart.
        """
        held_id, holder_id = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (self.held, self.holder)
        )
        distance = float(np.linalg.norm(data.xpos[held_id] - data.xpos[holder_id]))
        alignment = abs(float(data.xquat[held_id] @ data.xquat[holder_id]))
        return distance, 2.0 * float(np.arccos(min(1.0, alignment)))

    def is_breaking(self, *, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """
        :param model: The compiled scene holding the weld.
        :param data: Its simulation state after a step.
        :return: Whether the weld carries more than it can, or has been stretched
            apart.
        """
        load = self.load(model=model, data=data)
        distance, angle = self.drift(model=model, data=data)
        return bool(
            np.linalg.norm(load[:3]) > self.break_force
            or np.linalg.norm(load[3:]) > self.break_torque
            or distance > self.break_distance
            or angle > self.break_angle
        )
