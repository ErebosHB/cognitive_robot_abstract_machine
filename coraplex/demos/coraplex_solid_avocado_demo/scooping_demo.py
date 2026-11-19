"""
A spoon takes one scoop of flesh out of a half of the solid avocado held by two poles,
after a knife has halved it round the pit and taken the pit out. Whatever flesh its bowl
held on its way down and up again comes away on it and is laid on the board; the rest
stays in the skin.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np

from cutting import BodyPose, FruitMaterial, FruitPiece, SpoonBowl
from exceptions import FruitNotCut
from held_avocado_demonstration import HalvingAvocadoDemonstration
from solid_avocado_demonstration import SolidAvocadoDemonstration

# %% names and outcome


class SpoonName(StrEnum):
    """
    The named things of the spoon.
    """

    SPOON = "spoon"
    """
    The mocap body of the spoon, its frame in the middle of the bowl: the bowl's x
    axis points to its tip and its z axis out of its back.
    """


@dataclass
class ScoopingOutcome:
    """
    What the demonstration achieved.
    """

    scooped_flesh: float
    """
    The volume of the flesh the spoon took out, in cubic metres.
    """

    flesh_left: float
    """
    The volume of the flesh left in the half on the poles, in cubic metres.
    """

    scoop_height: float
    """
    How high above the board the scooped flesh's centre of mass rests, in metres.
    """

    spoon_stopped: bool
    """
    Whether skin or pit stopped the spoon on its way, too far off it to slide along
    them, so that it was drawn back out.
    """


# %% the way of the spoon


@dataclass
class ScoopArc:
    """
    A circle in a plane upright through the avocado's long axis, which the spoon is
    led round to scoop.
    """

    centre: np.ndarray
    """
    The middle of the circle, in the world, in metres.
    """

    up: np.ndarray
    """
    The direction out of the half's cut face, in the world.
    """

    towards_stem: np.ndarray
    """
    The direction of the avocado's long axis towards the stem, in the world.
    """

    radius: float
    """
    The radius of the circle in metres.
    """

    tilt: float
    """
    How far the spoon turns as it goes round the circle, as a share of how far round
    it goes; at 1 its bowl's back always faces away from the centre.
    """

    def pose_at(self, angle: float, distance: float) -> tuple[np.ndarray, np.ndarray]:
        """
        :param angle: How far round from the lowest point of the circle towards the
            stem, in radians.
        :param distance: How far from the centre the spoon's frame is, in metres.
        :return: The position and orientation, scalar first, of the spoon's frame
            there, its bowl's tip pointing towards the base, :attr:`tilt` of the way
            round the circle turned with it.
        """
        outward = np.sin(angle) * self.towards_stem - np.cos(angle) * self.up
        turned = self.tilt * angle
        back = np.sin(turned) * self.towards_stem - np.cos(turned) * self.up
        tip = -(np.cos(turned) * self.towards_stem + np.sin(turned) * self.up)
        orientation = np.column_stack([tip, np.cross(back, tip), back])
        return (
            self.centre + distance * outward,
            SolidAvocadoDemonstration.quaternion_of(orientation),
        )


@dataclass
class SpoonPass:
    """
    What has happened so far as the spoon is led along its way through the flesh.
    """

    scooped: FruitPiece | None = None
    """
    The flesh that has come away on the spoon's bowl, if any.
    """

    on_bowl: PieceOnTool | None = None
    """
    Where the scooped flesh lies relative to the spoon, once it has come away.
    """

    stopped: bool = False
    """
    Whether skin or pit has stopped the spoon.
    """


@dataclass
class PieceOnTool:
    """
    Where a piece carried on a tool lies, relative to the tool.
    """

    rotation: np.ndarray
    """
    The orientation of the piece's body in the tool's axes.
    """

    offset: np.ndarray
    """
    The position of the piece's body in the tool's frame, in metres.
    """

    @classmethod
    def between(cls, tool: BodyPose, piece: BodyPose) -> PieceOnTool:
        """
        :param tool: Where the tool is.
        :param piece: Where the piece's body is.
        :return: Where the piece lies relative to the tool.
        """
        return cls(
            tool.rotation().T @ piece.rotation(),
            tool.rotation().T @ (piece.position - tool.position),
        )

    def piece_pose(self, tool: BodyPose) -> tuple[np.ndarray, np.ndarray]:
        """
        :param tool: Where the tool is now.
        :return: The position and orientation, scalar first, of the piece's body.
        """
        return (
            tool.position + tool.rotation() @ self.offset,
            SolidAvocadoDemonstration.quaternion_of(tool.rotation() @ self.rotation),
        )


# %% the demonstration


@dataclass
class AvocadoScoopingDemonstration(HalvingAvocadoDemonstration):
    """
    The knife halves the held avocado and a spoon takes one scoop of flesh out of the
    half the poles hold.
    """

    bowl_size: tuple[float, float, float] = (0.016, 0.011, 0.003)
    """
    Half the length, width and thickness of the spoon's bowl, in metres.
    """

    spoon_speed: float = 0.03
    """
    How fast the spoon runs through the flesh, in metres per second.
    """

    scoop_position: float = 0.07
    """
    Where along the avocado's long axis, measured from its base, the spoon is led
    round, in metres: at the end of the pit's hollow, short of the pole there.
    """

    scoop_radius: float = 0.016
    """
    The radius of the circle the middle of the spoon's bowl is led round, in metres.
    """

    bowl_reach: float = 0.02
    """
    How far beyond its back, on its hollow side, the spoon's bowl holds what it
    scoops, in metres.
    """

    scoop_rise: float = 0.004
    """
    How far above the half's cut face the middle of the spoon's circle lies, in
    metres; the lower, the deeper the spoon goes.
    """

    spoon_tilt: float = 0.5
    """
    How far the spoon turns as it is led round its circle, as a share of how far
    round it goes, keeping its handle above the avocado.
    """

    slide_step: float = 0.0005
    """
    The steps, in metres, in which the spoon's bowl is slid off skin and pit in its
    way, and back towards its way once clear of them.
    """

    slide_limit: float = 0.015
    """
    How far in metres the spoon's bowl slides off its way at most before skin or pit
    stops it.
    """

    entry_overrun: float = 0.2
    """
    How far round the circle, in radians, above the cut face the spoon starts and
    ends.
    """

    spoon_dip: float = 0.05
    """
    How far above where it goes into the flesh the spoon comes down from and goes
    back up to, in metres.
    """

    scoop_place: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.09, 0.0]))
    """
    Where on the board the scooped flesh is laid, in the world, in metres.
    """

    def add_equipment(self, spec: mujoco.MjSpec) -> None:
        """
        Add the poles and the spoon, a mocap body that touches nothing, far above the
        board; see :meth:`HeldAvocadoDemonstration.add_equipment`.
        """
        super().add_equipment(spec)
        length, width, thickness = self.bowl_size
        spoon = spec.worldbody.add_body(
            name=SpoonName.SPOON, mocap=True, pos=[0.15, -0.3, 0.4]
        )
        steel = [0.78, 0.8, 0.83, 1]
        spoon.add_geom(
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=[length, width, thickness],
            rgba=steel,
            contype=0,
            conaffinity=0,
        )
        spoon.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[-length, 0, -thickness, -length - 0.09, 0, -0.07],
            size=[0.004, 0, 0],
            rgba=steel,
            contype=0,
            conaffinity=0,
        )

    def run(self) -> ScoopingOutcome:
        """
        Halve the avocado and take one scoop of flesh out of the held half.

        :return: What the demonstration achieved.
        """
        with self.shown():
            halves = self.halve()
            self.put_knife_away()
            spoon_pass = self.scoop(halves.lower_half)
            scooped = spoon_pass.scooped
            self.lift_out(scooped)
            flesh_left = sum(
                piece.part_of(FruitMaterial.FLESH).solid.volume()
                for piece in self.fruit.pieces
                if self.fruit.is_held(piece)
                and FruitMaterial.FLESH in piece.materials()
            )
        return ScoopingOutcome(
            scooped.volume(),
            flesh_left,
            float(self.centre_of(scooped)[2] - self.board_size[2]),
            spoon_pass.stopped,
        )

    def scoop_arc(self) -> ScoopArc:
        """
        :return: The circle the spoon follows through the held half.
        """
        face = self.halving.plane_in(self.whole_fruit)
        towards_stem = self.whole_fruit.rotation() @ np.array([0.0, 0.0, 1.0])
        middle = self.whole_fruit.to_world(np.array([[0.0, 0.0, self.scoop_position]]))[
            0
        ]
        centre = middle - ((middle - face.point) @ face.normal) * face.normal
        return ScoopArc(
            centre + self.scoop_rise * face.normal,
            face.normal,
            towards_stem,
            self.scoop_radius,
            self.spoon_tilt,
        )

    def scoop(self, half: FruitPiece) -> SpoonPass:
        """
        Lead the spoon round :meth:`scoop_arc`, from above the held half's cut face on
        the stem's side down into the flesh and up again, and carry on with whatever
        flesh comes away on its bowl. The bowl slides along skin and pit in its way;
        where they stop it, the spoon is drawn back out the way it came.

        :param half: The held half, cut face up.
        :return: What happened on the spoon's way, the flesh scooped among it.
        :raises FruitNotCut: If the spoon's way scoops no flesh out.
        """
        arc = self.scoop_arc()
        bowl = SpoonBowl(SpoonName.SPOON, self.bowl_size, self.bowl_reach)
        # Where the circle crosses the cut face; nowhere if it lies wholly above it.
        crossing = np.arccos(np.clip(self.scoop_rise / arc.radius, -1.0, 1.0))
        entry = crossing + self.entry_overrun
        step_angle = self.spoon_speed * self.fruit.model.opt.timestep / arc.radius
        way = [
            arc.pose_at(angle, arc.radius)
            for angle in np.arange(entry, -entry, -step_angle)
        ]
        position, quaternion = way[0]
        self.glide(
            SpoonName.SPOON,
            position + self.spoon_dip * arc.up,
            quaternion,
            self.tool_speed,
        )
        self.glide(SpoonName.SPOON, position, quaternion, self.spoon_speed)
        spoon_pass = SpoonPass()
        reached = self.lead_spoon(way[1:], bowl, half, spoon_pass)
        if spoon_pass.stopped:
            self.lead_spoon(reached[::-1], bowl, half, spoon_pass)
        if spoon_pass.scooped is None:
            piece_count = len(self.fruit.pieces)
            raise FruitNotCut(piece_count + 1, piece_count)
        return spoon_pass

    def lead_spoon(
        self,
        way: list[tuple[np.ndarray, np.ndarray]],
        bowl: SpoonBowl,
        half: FruitPiece,
        spoon_pass: SpoonPass,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Move the spoon through poses, one per step, sliding its bowl along skin and pit
        in its way, until they would stop it; scoop out of a half, then carry what
        comes away.

        :param way: The positions and orientations, scalar first, of the spoon.
        :param bowl: The spoon's bowl.
        :param half: The half it scoops.
        :param spoon_pass: What has happened on the spoon's way so far; updated.
        :return: The poses the spoon reached, slid along skin and pit.
        """
        flesh = frozenset({FruitMaterial.FLESH})
        reached, offset = [], 0.0
        for position, quaternion in way:
            offset = self.slide_offset(bowl, position, quaternion, offset)
            if offset is None:
                spoon_pass.stopped = True
                return reached
            pose = BodyPose(position, quaternion, np.zeros(6))
            pose.position = position - offset * pose.rotation()[:, 2]
            reached.append((pose.position, quaternion))
            self.move_mocap(SpoonName.SPOON, pose.position, quaternion)
            if spoon_pass.on_bowl is not None:
                self.move_mocap(
                    spoon_pass.scooped.body_name, *spoon_pass.on_bowl.piece_pose(pose)
                )
            self.step()
            if spoon_pass.scooped is not None:
                continue
            made = self.fruit.scoop_with(bowl, [half], flesh)
            if made:
                spoon_pass.scooped = made[0]
                self.fruit.grip(spoon_pass.scooped)
                self.show_model()
                spoon_pass.on_bowl = PieceOnTool.between(
                    pose, self.fruit.pose_of(spoon_pass.scooped)
                )
        return reached

    def slide_offset(
        self,
        bowl: SpoonBowl,
        position: np.ndarray,
        quaternion: np.ndarray,
        last_offset: float,
    ) -> float | None:
        """
        :param bowl: The spoon's bowl.
        :param position: Where the spoon is to go, in the world, in metres.
        :param quaternion: Its orientation there, scalar first.
        :param last_offset: How far the bowl was slid at the pose before, in metres.
        :return: How far, at least :attr:`slide_step` less than *last_offset*, the bowl
            has to be slid towards its hollow side to clear skin and pit there, in
            metres; None if farther than :attr:`slide_limit`.
        """
        pose = BodyPose(position, quaternion, np.zeros(6))
        hollow_side = -pose.rotation()[:, 2]
        flesh = frozenset({FruitMaterial.FLESH})
        for offset in np.arange(
            max(0.0, last_offset - self.slide_step), self.slide_limit, self.slide_step
        ):
            slid = BodyPose(position + offset * hollow_side, quaternion, np.zeros(6))
            if not self.fruit.bowl_blocked(bowl, slid, flesh):
                return float(offset)
        return None

    def spoon_pose(self) -> BodyPose:
        """
        :return: Where the spoon is.
        """
        mocap_id = self.mocap_id(SpoonName.SPOON)
        data = self.fruit.data
        return BodyPose(
            data.mocap_pos[mocap_id].copy(),
            data.mocap_quat[mocap_id].copy(),
            np.zeros(6),
        )

    def lift_out(self, scooped: FruitPiece) -> None:
        """
        Lift the scooped flesh out on the spoon, lay it on the board and draw the spoon
        back up.

        :param scooped: The scooped flesh, on the spoon's bowl.
        """
        # Laid down as it lay on the bowl, the side away from the bowl up.
        spoon, piece = self.spoon_pose(), self.fruit.pose_of(scooped)
        off_bowl = piece.rotation().T @ spoon.rotation() @ np.array([0.0, 0.0, -1.0])
        self.lay_aside(
            scooped,
            self.scoop_place - self.centre_of(scooped),
            off_bowl,
            SpoonName.SPOON,
        )
        self.let_pieces_rest()
        mocap_id = self.mocap_id(SpoonName.SPOON)
        self.glide(
            SpoonName.SPOON,
            self.fruit.data.mocap_pos[mocap_id] + np.array([0.0, 0.0, self.spoon_dip]),
            self.fruit.data.mocap_quat[mocap_id].copy(),
            self.tool_speed,
        )


# %% entry point


def main(
    headless: bool = False, playback_speed: float = 1.0, pit_seed: int | None = None
) -> ScoopingOutcome:
    """
    Run the demonstration and report what it achieved.

    :param headless: Whether the physics runs without a viewer.
    :param playback_speed: How many seconds of simulated time the viewer shows per
        second of wall-clock time.
    :param pit_seed: The seed of the avocado's random pit; None draws one.
    :return: What the demonstration achieved.
    """
    demonstration = AvocadoScoopingDemonstration(
        headless=headless, playback_speed=playback_speed, pit_seed=pit_seed
    )
    outcome = demonstration.run()
    print(f"The pit came from seed {demonstration.pit_seed}.")
    print(
        f"Scooped {outcome.scooped_flesh * 1e6:.1f} cm³ of flesh out of the held half; "
        f"it rests {outcome.scoop_height * 1000:.0f} mm above the board, and "
        f"{outcome.flesh_left * 1e6:.1f} cm³ of flesh stay in the half on the poles."
        + (
            " Skin or pit stopped the spoon, so it was drawn back out."
            if outcome.spoon_stopped
            else ""
        )
    )
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take one scoop of flesh out of a half of a solid avocado held by "
        "two poles in MuJoCo."
    )
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="simulated seconds shown per second; 1 is real time, 0.5 half speed",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="the seed of the random pit"
    )
    arguments = parser.parse_args()
    main(arguments.headless, arguments.speed, arguments.seed)
