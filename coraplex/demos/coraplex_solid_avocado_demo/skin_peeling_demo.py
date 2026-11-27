"""
The solid avocado, held by two poles, is peeled: a knife scores its skin lengthwise into
strips, cutting through the skin only, and every strip the poles do not pierce is peeled
off the flesh from its end, bending as it comes off, and laid along the board.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from cutting import BodyPose, CuttableFruit, CuttingPlane, FruitPiece
from fruit_material import FruitMaterial
from exceptions import FruitNotCut
from held_avocado_demonstration import HeldAvocadoDemonstration
from knife_strokes import ScoringStroke
from solid_avocado_demonstration import SolidAvocadoDemonstration

# %% outcome


@dataclass
class PeelingOutcome:
    """
    What the demonstration achieved.
    """

    peeled_skin: float
    """
    The volume of the strips of skin pulled off, in cubic metres.
    """

    skin_left: float
    """
    The volume of skin left on the avocado where the poles pierce it, in cubic metres.
    """

    peeled_strips: int
    """
    How many strips were pulled off.
    """

    largest_deviation: float
    """
    The farthest in metres any cut surface lay from the plane of the blade that made it.
    """


# %% the demonstration


@dataclass
class SkinPeelingDemonstration(HeldAvocadoDemonstration):
    """
    The knife scores the held avocado's skin into strips that are then peeled off.
    """

    strip_count: int = 8
    """
    How many strips the skin is scored into, an even number: each stroke scores it along
    two opposite lines, all at equal angles round the long axis, none above a pole.
    """

    scoring_depth: float = 0.003
    """
    How far under the surface the cutting edge scores, in metres: deeper than the skin
    is thick, as the edge is straight and the avocado's outline curves.
    """

    hinge_spacing: float = 0.012
    """
    The distance along the long axis between the lines a peeled strip bends along, in
    metres.
    """

    peel_duration: float = 3.0
    """
    How long peeling a strip off, up to the segment at the base, takes, in seconds.
    """

    strip_column: float = -0.19
    """
    Where along the board the end of each strip laid down that was at the base lies, in
    metres.
    """

    strip_spacing: float = 0.038
    """
    The distance between neighbouring strips laid on the board, in metres.
    """

    def run(self) -> PeelingOutcome:
        """
        Score the skin into strips and peel off the strips the poles do not pierce.

        :return: What the demonstration achieved.
        """
        with self.shown():
            strips = self.score()
            on_poles = [
                strip
                for strip in strips
                if self.fruit.reaches_holder(strip, self.fruit.pose_of(strip))
            ]
            loose = [strip for strip in strips if strip not in on_poles]
            self.peel(loose)
        return PeelingOutcome(
            sum(strip.volume() for strip in loose),
            sum(strip.volume() for strip in on_poles),
            len(loose),
            self.largest_deviation,
        )

    def scoring_strokes(self) -> list[ScoringStroke]:
        """
        :return: The strokes that score the skin, each in a plane through the long axis.
        """
        middle = np.array([0.0, 0.0, self.fruit_length / 2])
        strokes = []
        for number in range(self.strip_count // 2):
            # Angles round the long axis start at the frame's x axis, pointing down.
            angle = (number + 0.5) * 2 * np.pi / self.strip_count
            strokes.append(
                ScoringStroke(
                    middle,
                    np.array([-np.sin(angle), np.cos(angle), 0.0]),
                    approach=np.array([-1.0, 0.0, 0.0]),
                    depth=self.scoring_depth,
                )
            )
        return strokes

    def score(self) -> list[FruitPiece]:
        """
        Score the skin along every stroke and part the strip on the flesh from it.

        :return: The strips of skin, each still on the flesh.
        :raises FruitNotCut: If a stroke does not score the skin all the way round.
        """
        fruit = self.fruit
        for number, stroke in enumerate(self.scoring_strokes()):
            self.cut_along(stroke)
            expected_pieces = 2 * (number + 1)
            if len(fruit.pieces) != expected_pieces:
                raise FruitNotCut(expected_pieces, len(fruit.pieces))
        self.put_knife_away()
        fruit.separate(fruit.piece_with(FruitMaterial.FLESH), FruitMaterial.SKIN)
        self.show_model()
        return [
            piece for piece in fruit.pieces if piece.materials() == {FruitMaterial.SKIN}
        ]

    def peel(self, strips: list[FruitPiece]) -> None:
        """
        Peel each strip off from its end at the stem, bending it straight, and lay it
        flat along the board inside up, the strips side by side in the order they lie
        round the avocado.

        :param strips: The strips.
        """
        ordered = sorted(strips, key=lambda strip: self.outward_of(strip)[1])
        first = -self.strip_spacing * (len(ordered) - 1) / 2
        for number, strip in enumerate(ordered):
            outward = self.outward_of(strip)
            segments = self.fruit.hinge(strip, self.hinge_planes(strip))
            for segment in segments:
                self.fruit.grip(segment)
            self.show_model()
            hinged = HingedStrip.of(self.fruit, segments)
            self.peel_off(hinged)
            self.lay_flat(hinged, outward, first + number * self.strip_spacing)
            for segment in segments:
                self.released(segment)
            self.simulate(self.rest_duration)
            self.let_pieces_rest()

    def outward_of(self, strip: FruitPiece) -> np.ndarray:
        """
        :param strip: A strip of skin on the avocado.
        :return: The direction in the world from the long axis out to the strip.
        """
        axis = self.whole_fruit.rotation() @ np.array([0.0, 0.0, 1.0])
        offset = self.centre_of(strip) - self.whole_fruit.position
        radial = offset - (offset @ axis) * axis
        return radial / np.linalg.norm(radial)

    def hinge_planes(self, strip: FruitPiece) -> list[CuttingPlane]:
        """
        :param strip: A strip of skin on the avocado.
        :return: The planes across the long axis, :attr:`hinge_spacing` apart, along
            which the strip bends.
        """
        axis = self.whole_fruit.rotation() @ np.array([0.0, 0.0, 1.0])
        reach = (
            self.fruit.pose_of(strip).to_world(strip.outline.vertices)
            - self.whole_fruit.position
        ) @ axis
        return [
            CuttingPlane(self.whole_fruit.position + position * axis, axis)
            for position in np.arange(
                reach.min() + self.hinge_spacing, reach.max(), self.hinge_spacing
            )
        ]

    def peel_off(self, hinged: HingedStrip) -> None:
        """
        Straighten the hinges of a gripped strip one after another from the stem towards
        the base, so the strip comes off the flesh from its end, until only the segment
        at the base still lies on it.

        :param hinged: The strip.
        """
        hinge_count = len(hinged.flat_angles)
        steps = round(self.peel_duration / self.fruit.model.opt.timestep)
        # Counted from the hinge at the stem.
        order = hinge_count - 1 - np.arange(hinge_count)
        for step in range(1, steps + 1):
            # The hinges straighten in turn, each over the time of two, overlapping the
            # next.
            elapsed = step / steps * hinge_count
            straightened = np.clip((elapsed - order) / 2, 0.0, 1.0)
            straightened[0] = 0.0
            self.move_segments(hinged, straightened, np.eye(3), np.zeros(3))
            self.step()

    def lay_flat(self, hinged: HingedStrip, outward: np.ndarray, row: float) -> None:
        """
        Lift a gripped strip off the avocado, straightening its last hinge, and carry it
        over to lie flat along the board, inside up, just above it.

        :param hinged: The strip, peeled off up to the segment at the base.
        :param outward: The direction in the world from the long axis out to the strip
            on the avocado.
        :param row: Where across the board the strip is laid, in metres.
        """
        straight = np.ones(len(hinged.flat_angles))
        corners = hinged.corners(straight)
        along = hinged.chords()[0]
        outside = outward - (outward @ along) * along
        outside /= np.linalg.norm(outside)
        # Along the strip to the board's x axis and its outside facing down.
        rotation = (
            np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
            @ np.column_stack([along, np.cross(outside, along), outside]).T
        )
        laid = corners @ rotation.T
        lowest, highest = laid.min(axis=0), laid.max(axis=0)
        target = np.array(
            [self.strip_column, row, self.board_size[2] + self.drop_height]
        ) - np.array([lowest[0], (lowest[1] + highest[1]) / 2, lowest[2]])
        centre = corners.mean(axis=0)
        end = rotation @ centre + target
        axis, angle = self.rotation_between(
            np.array([1.0, 0.0, 0.0, 0.0]), self.quaternion_of(rotation)
        )
        carried = max(centre[2], end[2]) + self.lift_height
        steps = round(self.lay_aside_duration / self.fruit.model.opt.timestep)
        for step in range(1, steps + 1):
            position, across = self.carried_position(centre, end, carried, step / steps)
            turning = self.turn(axis, across * angle)[1]
            straightened = straight.copy()
            straightened[0] = across
            self.move_segments(
                hinged, straightened, turning, position - turning @ centre
            )
            self.step()

    def move_segments(
        self,
        hinged: HingedStrip,
        straightened: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        """
        Put the gripped segments of a strip where straightening its hinges by given
        amounts, then moving the whole strip, puts them.

        :param hinged: The strip.
        :param straightened: How far each hinge is straightened, from 0 for not at all
            to 1 for fully.
        :param rotation: How the whole strip is turned afterwards, in the world.
        :param translation: How far it is moved after turning, in the world, in metres.
        """
        start = hinged.body_pose
        for segment, (turned, moved) in zip(
            hinged.segments, hinged.placements(straightened)
        ):
            self.move_mocap(
                segment.body_name,
                rotation @ (turned @ start.position + moved) + translation,
                self.quaternion_of(rotation @ turned @ start.rotation()),
            )


# %% a strip that bends


@dataclass
class HingedStrip:
    """
    A strip of skin cut into segments, each joined to the next by a hinge, as it lay
    when it was cut; straightening its hinges lays its segments in a line.
    """

    segments: list[FruitPiece]
    """
    The segments, from the base to the stem; the first stays where it is.
    """

    body_pose: BodyPose
    """
    Where the bodies of all segments were when the strip was cut into them.
    """

    hinge_ends: list[list[np.ndarray]]
    """
    The two ends of the line each hinge turns about, in the world, in metres, from the
    hinge at the base to the one at the stem.
    """

    centres: list[np.ndarray]
    """
    The centre of mass of each segment, in the world, in metres.
    """

    flat_angles: np.ndarray = field(init=False)
    """
    How far, in radians, each hinge turns the segments beyond it to bring them in line
    with the segment before it.
    """

    def __post_init__(self) -> None:
        """
        Work out :attr:`flat_angles`.
        """
        chords = self.chords()
        angles = []
        for (start, end), behind, ahead in zip(self.hinge_ends, chords, chords[1:]):
            axis = (end - start) / np.linalg.norm(end - start)
            behind = behind - (behind @ axis) * axis
            ahead = ahead - (ahead @ axis) * axis
            angles.append(np.arctan2(axis @ np.cross(ahead, behind), ahead @ behind))
        self.flat_angles = np.array(angles)

    @classmethod
    def of(cls, fruit: CuttableFruit, segments: list[FruitPiece]) -> HingedStrip:
        """
        :param fruit: The fruit the strip is a piece of.
        :param segments: The segments :meth:`CuttableFruit.hinge` cut the strip into,
            still where they were cut.
        :return: The strip.
        """
        pose = fruit.pose_of(segments[0])
        return cls(
            segments,
            pose,
            [fruit.hinge_ends(first, second) for first, second in pairwise(segments)],
            [
                pose.to_world(segment.mass_distribution().centre[None])[0]
                for segment in segments
            ],
        )

    def chords(self) -> np.ndarray:
        """
        :return: For each segment, the direction from where it begins to where it ends:
            its centre at either end of the strip, else the middles of its hinges.
        """
        middles = [(start + end) / 2 for start, end in self.hinge_ends]
        chords = np.diff(
            np.array([self.centres[0], *middles, self.centres[-1]]), axis=0
        )
        return chords / np.linalg.norm(chords, axis=1, keepdims=True)

    def placements(
        self, straightened: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        :param straightened: How far each hinge is straightened, from 0 for not at all
            to 1 for fully.
        :return: For each segment, the rotation and then the translation, in the world,
            in metres, that move it from where it was cut to where it is then.
        """
        turned, moved = np.eye(3), np.zeros(3)
        placements = [(turned, moved)]
        for (start, end), fraction, angle in zip(
            self.hinge_ends, straightened, self.flat_angles
        ):
            axis = (end - start) / np.linalg.norm(end - start)
            matrix = SolidAvocadoDemonstration.turn(axis, fraction * angle)[1]
            # Each hinge turns about its line as it lay, after the hinges before it.
            moved = turned @ (start - matrix @ start) + moved
            turned = turned @ matrix
            placements.append((turned, moved))
        return placements

    def corners(self, straightened: np.ndarray) -> np.ndarray:
        """
        :param straightened: How far each hinge is straightened.
        :return: The corners of the segments' surfaces then, in the world, in metres,
            one row each.
        """
        return np.vstack(
            [
                self.body_pose.to_world(segment.outline.vertices) @ turned.T + moved
                for segment, (turned, moved) in zip(
                    self.segments, self.placements(straightened)
                )
            ]
        )


# %% entry point


def main(
    headless: bool = False,
    playback_speed: float = 1.0,
    pit_seed: int | None = None,
    firmness: float | None = None,
) -> PeelingOutcome:
    """
    Run the demonstration and report what it achieved.

    :param headless: Whether the physics runs without a viewer.
    :param playback_speed: How many seconds of simulated time the viewer shows per
        second of wall-clock time.
    :param pit_seed: The seed of the avocado's random pit; None draws one.
    :param firmness: How firm the avocado is, in newtons; None draws a firmness from the
        seed.
    :return: What the demonstration achieved.
    """
    demonstration = SkinPeelingDemonstration(
        headless=headless,
        playback_speed=playback_speed,
        pit_seed=pit_seed,
        firmness=firmness,
    )
    outcome = demonstration.run()
    print(f"The pit came from seed {demonstration.pit_seed}.")
    print(demonstration.texture_report())
    print(
        f"Peeled {outcome.peeled_strips} strips of skin, "
        f"{outcome.peeled_skin * 1e6:.1f} cm³, off the held avocado; "
        f"{outcome.skin_left * 1e6:.1f} cm³ stay where the poles pierce it. Every "
        f"score lies within {outcome.largest_deviation * 1e9:.0f} nm of the blade's "
        f"plane."
    )
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Peel a solid avocado held by two poles in MuJoCo."
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
    parser.add_argument(
        "--firmness",
        type=float,
        default=None,
        help="how firm the avocado is in newtons, from 200 (hard) to 7 (overripe)",
    )
    arguments = parser.parse_args()
    main(arguments.headless, arguments.speed, arguments.seed, arguments.firmness)
