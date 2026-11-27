"""
A knife cuts a solid avocado lying on a cutting board: a slanted slice off the stem end,
then in half lengthwise round the pit, which no knife cuts.

The pieces are laid aside and the pit is taken out. The avocado is one body of skin,
flesh and pit until the blade has gone all the way through it; see :mod:`cutting`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import mujoco
import numpy as np

from fruit_material import FruitMaterial
from exceptions import FruitNotCut
from knife_strokes import DownwardStroke, KnifeStroke, StrokeAroundPit
from solid_avocado_demonstration import SolidAvocadoDemonstration

# %% steps and outcome


@dataclass
class CuttingStep:
    """
    A stroke of the knife and what is done with the piece it cuts off.
    """

    stroke: KnifeStroke
    """
    The stroke.
    """

    expected_pieces: int
    """
    How many pieces the fruit is in after the stroke.
    """

    set_aside: np.ndarray
    """
    How far the piece cut off on the side the plane's normal points to is laid aside,
    in the frame of the whole fruit, in metres; only its horizontal part counts.
    """

    face_up: bool
    """
    Whether that piece is laid down cut face up, which only a piece that can rest so
    should be; otherwise it is carried the way it lies.
    """


@dataclass
class CuttingOutcome:
    """
    What the demonstration achieved.
    """

    piece_volumes: list[float]
    """
    The volume of each piece the fruit ended in, in cubic metres.
    """

    largest_deviation: float
    """
    The farthest in metres any cut surface lay from the plane of the blade that made it.
    """

    pit_piece: str
    """
    The name of the body of the pit, taken out whole.
    """


# %% the demonstration


@dataclass
class AvocadoCuttingDemonstration(SolidAvocadoDemonstration):
    """
    The knife cuts the solid avocado lying free on the cutting board.
    """

    steps: list[CuttingStep] = field(
        default_factory=lambda: [
            CuttingStep(
                DownwardStroke(
                    np.array([0.0, 0.0, 0.085]), np.array([0.25, 0.35, 1.0])
                ),
                2,
                set_aside=np.array([0.0, 0.0, 0.05]),
                face_up=False,
            ),
            CuttingStep(
                StrokeAroundPit(
                    np.array([0.003, 0.002, 0.0485]),
                    np.array([-1.0, 0.0, 0.0]),
                    approach=np.array([0.0, -1.0, 0.0]),
                ),
                3,
                set_aside=np.array([0.0, 0.085, 0.0]),
                face_up=True,
            ),
        ]
    )
    """
    The steps, in order.

    The avocado lies on its side, its frame's x axis pointing down; the second stroke
    halves it level, through the pit's middle.
    """

    tools_push_pieces: bool = False
    """
    See :attr:`SolidAvocadoDemonstration.tools_push_pieces`; off, as nothing holds the
    avocado lying free on the board, and the knife would push it away from its cut.
    """

    pit_set_aside: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.09])
    )
    """
    How far the pit is laid aside when it is taken out, in the frame of the whole fruit,
    in metres; only its horizontal part counts.
    """

    def place_avocado(self, avocado: mujoco.MjsBody) -> None:
        """
        Lay the avocado on its side on the board.
        """
        avocado.pos = [-0.055, 0, self.board_size[2] + 0.036]
        avocado.quat = [np.sqrt(0.5), 0, np.sqrt(0.5), 0]

    def run(self) -> CuttingOutcome:
        """
        Make every step and take the pit out.

        :return: What the demonstration achieved.
        """
        with self.shown():
            whole_fruit = self.whole_fruit
            for step in self.steps:
                cut_off = self.cut_along(step.stroke)
                if len(self.fruit.pieces) != step.expected_pieces:
                    raise FruitNotCut(step.expected_pieces, len(self.fruit.pieces))
                self.let_pieces_rest()
                offset = whole_fruit.rotation() @ step.set_aside
                for entry in cut_off:
                    if not entry.on_normal_side:
                        continue
                    face_normal = entry.face_normal if step.face_up else None
                    self.lay_aside(entry.piece, offset, face_normal)
                self.let_pieces_rest()
            pit = self.fruit.separate(
                self.fruit.piece_with(FruitMaterial.PIT), FruitMaterial.PIT
            )
            self.show_model()
            self.lay_aside(pit, whole_fruit.rotation() @ self.pit_set_aside, None)
            self.let_pieces_rest()
        return CuttingOutcome(
            [piece.volume() for piece in self.fruit.pieces],
            self.largest_deviation,
            pit.body_name,
        )


# %% entry point


def main(
    headless: bool = False,
    playback_speed: float = 1.0,
    pit_seed: int | None = None,
    firmness: float | None = None,
) -> CuttingOutcome:
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
    demonstration = AvocadoCuttingDemonstration(
        headless=headless,
        playback_speed=playback_speed,
        pit_seed=pit_seed,
        firmness=firmness,
    )
    outcome = demonstration.run()
    print(f"The pit came from seed {demonstration.pit_seed}.")
    print(demonstration.texture_report())
    volumes = ", ".join(f"{volume * 1e6:.0f}" for volume in outcome.piece_volumes)
    print(
        f"Cut the avocado into pieces of {volumes} cm³, every cut within "
        f"{outcome.largest_deviation * 1e9:.0f} nm of the blade's plane, and took the "
        f"pit out as {outcome.pit_piece}."
    )
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut a solid avocado with a knife in MuJoCo."
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
