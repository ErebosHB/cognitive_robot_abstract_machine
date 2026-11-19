"""
A knife cuts the solid avocado, held by two poles, into slices: it halves the avocado
level round the pit, the upper half is laid aside and the pit taken out, and the held
half is sliced crosswise, its slices falling onto the board. Then the upper half is
pushed onto the poles in its place and sliced the same way.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from cutting import FruitPiece
from exceptions import FruitNotCut
from held_avocado_demonstration import HalvingAvocadoDemonstration
from knife_strokes import DownwardStroke

# %% outcome


@dataclass
class SlicingOutcome:
    """
    What the demonstration achieved.
    """

    slice_volumes: list[float]
    """
    The volume of each slice, in cubic metres.
    """

    largest_deviation: float
    """
    The farthest in metres any cut surface lay from the plane of the blade that made it.
    """


# %% the demonstration


@dataclass
class AvocadoSlicingDemonstration(HalvingAvocadoDemonstration):
    """
    The knife halves the held avocado and cuts both halves into slices.
    """

    slice_positions: list[float] = field(
        default_factory=lambda: [0.026, 0.034, 0.042, 0.050, 0.058, 0.066, 0.074]
    )
    """
    Where along the avocado's long axis, measured from its base, the knife cuts each
    half crosswise, in metres; the slices lie between neighbouring cuts.
    """

    fall_duration: float = 1.0
    """
    How long a slice cut off the held half is left to fall onto the board, in seconds.
    """

    slice_rows: tuple[float, float] = (-0.075, 0.075)
    """
    Where across the board the slices of the lower and of the upper half are laid out,
    in metres.
    """

    end_places: tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (
            np.array([-0.155, 0.0, 0.0]),
            np.array([0.16, -0.09, 0.0]),
        )
    )
    """
    Where on the board the ends of the lower half, at the base and at the stem, are
    laid when they come off the poles, in the world, in metres.
    """

    slice_spacing: float = 0.045
    """
    The distance between neighbouring slices laid out in a row, in metres.
    """

    def run(self) -> SlicingOutcome:
        """
        Halve the avocado, slice both halves and lay the slices out.

        :return: What the demonstration achieved.
        """
        with self.shown():
            halves = self.halve()
            ends, lower_half_slices = self.slice(halves.lower_half, self.slice_rows[0])
            # Each end is laid on its crosswise cut face, which looks along the long
            # axis towards the other end.
            for end, place, top in zip(ends, self.end_places, [-1.0, 1.0]):
                offset = place - self.centre_of(end)
                self.lay_aside(end, offset, np.array([0.0, 0.0, top]))
                self.let_pieces_rest()
            self.push_onto_poles(halves.upper_half)
            _, upper_half_slices = self.slice(halves.upper_half, self.slice_rows[1])
            slices = lower_half_slices + upper_half_slices
        return SlicingOutcome(
            [piece.volume() for piece in slices], self.largest_deviation
        )

    def slice(
        self, half: FruitPiece, row: float
    ) -> tuple[list[FruitPiece], list[FruitPiece]]:
        """
        Cut a half held by the poles crosswise at each of :attr:`slice_positions`, from
        the base towards the stem, each stroke through what the one before left of it,
        and lay each slice that falls onto the board into a row before the next stroke.

        :param half: The half.
        :param row: Where across the board the row of slices lies, in metres.
        :return: The ends at the base and at the stem, which the poles still hold, and
            the slices between them, laid out in their row.
        :raises FruitNotCut: If a stroke does not cut through.
        """
        rest, ends, slices = half, [], []
        for position in self.slice_positions:
            stroke = DownwardStroke(
                np.array([0.0, 0.0, position]),
                np.array([0.0, 0.0, 1.0]),
                [rest],
            )
            expected_pieces = len(self.fruit.pieces) + 1
            cut_off = self.cut_along(stroke)
            if len(self.fruit.pieces) != expected_pieces:
                raise FruitNotCut(expected_pieces, len(self.fruit.pieces))
            self.simulate(self.fall_duration)
            self.let_pieces_rest()
            rest = next(entry.piece for entry in cut_off if entry.on_normal_side)
            behind = next(entry.piece for entry in cut_off if not entry.on_normal_side)
            if self.fruit.is_held(behind):
                ends.append(behind)
                continue
            self.lay_in_row(behind, len(slices), row)
            slices.append(behind)
        return [*ends, rest], slices

    def lay_in_row(self, piece: FruitPiece, number: int, row: float) -> None:
        """
        Lay a slice down flat in its place in a row along the board, its long side
        across the row.

        :param piece: The slice, cut crosswise to the avocado's long axis.
        :param number: Its place in the row, counted from 0 at the base's end.
        :param row: Where across the board the row lies, in metres.
        """
        # Every stroke but the first, which cuts off the held end at the base, cuts off
        # a slice.
        slice_count = len(self.slice_positions) - 1
        first = -self.slice_spacing * (slice_count - 1) / 2
        target = np.array([first + number * self.slice_spacing, row, 0.0])
        self.lay_aside(
            piece,
            target - self.centre_of(piece),
            np.array([0.0, 0.0, 1.0]),
            long_side=np.array([0.0, 1.0, 0.0]),
        )
        self.let_pieces_rest()

    def push_onto_poles(self, half: FruitPiece) -> None:
        """
        Carry a half lying cut face up on the board over the poles and push it down onto
        them, where the lower half lay, and hold it there.

        :param half: The half.
        """
        whole = self.whole_fruit
        along_axis = whole.rotation() @ np.array([0.0, 0.0, 1.0])
        quaternion, rotation = self.turn(along_axis, np.pi)
        quaternion = self.compose(quaternion, whole.quaternion)
        # Turning about the long axis, which runs through the body's origin.
        centre = whole.position + rotation @ whole.rotation() @ (
            half.mass_distribution().centre
        )
        self.carry(half, centre, quaternion)
        self.fruit.hold(half)
        self.show_model()
        self.let_pieces_rest()


# %% entry point


def main(
    headless: bool = False, playback_speed: float = 1.0, pit_seed: int | None = None
) -> SlicingOutcome:
    """
    Run the demonstration and report what it achieved.

    :param headless: Whether the physics runs without a viewer.
    :param playback_speed: How many seconds of simulated time the viewer shows per
        second of wall-clock time.
    :param pit_seed: The seed of the avocado's random pit; None draws one.
    :return: What the demonstration achieved.
    """
    demonstration = AvocadoSlicingDemonstration(
        headless=headless, playback_speed=playback_speed, pit_seed=pit_seed
    )
    outcome = demonstration.run()
    print(f"The pit came from seed {demonstration.pit_seed}.")
    volumes = ", ".join(f"{volume * 1e6:.1f}" for volume in outcome.slice_volumes)
    print(
        f"Cut the held avocado into {len(outcome.slice_volumes)} slices of {volumes} "
        f"cm³, every cut within {outcome.largest_deviation * 1e9:.0f} nm of the "
        f"blade's plane."
    )
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slice a solid avocado held by two poles in MuJoCo."
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
