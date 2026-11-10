"""
The PR2 puts an avocado on a cutting board in the kitchen of the IAI apartment and peels
one strip of its skin, in MuJoCo.

The avocado starts on the kitchen island beside the board. The PR2 stands at the island,
takes the avocado with its right hand from above and lays it on the board, stem to the
robot's right. Its left hand picks the fruit up by its base end and holds it out above
the board, turning it until a strip faces up. The right hand pinches that strip near
the held end, below the last rows at the base, where the strip would fork, peels it
off toward the stem and lays it on the board beside the fruit; the rest of the skin
stays on. ``pr2_whole_avocado_demo.py`` goes on to peel all of it.

The PR2 is the one of the semantic digital twin, exported to MuJoCo when the
demonstration starts; its meshes come from ``iai_pr2_description``. The avocado is seed 0
of the peelable avocado of ``private_meshes``, which comes with CRAM. Run it from this
folder::

    python pr2_demo.py

or pass another avocado's folder, or set ``AVOCADO_DIRECTORY`` to it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from demo import avocado_folder
from exceptions import SkinTornAtRest, StripNotPeeled
from pr2_whole_avocado_demo import Pr2WholeAvocadoPeelingDemonstration


@dataclass
class Pr2StripPeelingOutcome:
    """
    What the PR2 achieved in a run.
    """

    placement_error: float
    """
    How far in metres from the board's centre the avocado came to rest.
    """

    strip_length: int
    """
    How many tiles the strip held when it was laid down.
    """


@dataclass
class Pr2AvocadoPeelingDemonstration(Pr2WholeAvocadoPeelingDemonstration):
    """
    The PR2 puts the avocado from the kitchen island onto the cutting board and peels
    one strip of its skin, checking that each stage ends as it should.
    """

    pinched_width: int = 1
    """
    How many neighbouring tiles of one row the fingertips take between them: one, for a
    strip one column wide.
    """

    maximum_strips_per_end: int = 1
    """
    How many strips the right hand peels: one.
    """

    minimum_strip_tiles: int = 15
    """
    How many tiles, the pinched one included, must still hang together on the fingers
    when the strip is laid down: a strip, not tiles that snapped off one by one.
    """

    def run(self) -> Pr2StripPeelingOutcome:
        """
        :return: What the PR2 achieved.
        :raises StripNotPeeled: If the strip came apart or no strip was peeled.
        """
        self.build_scene()
        try:
            with self.viewer_context():
                torn_tiles = self.simulate(self.rest_duration)
                if torn_tiles:
                    raise SkinTornAtRest([t.description.body for t in torn_tiles])
                placement_error = self.put_avocado_on_board()
                self.pick_up_by_end(toward_stem=False)
                strip_lengths = self.peel_all_around() or [0]
                if strip_lengths[0] < self.minimum_strip_tiles:
                    raise StripNotPeeled(
                        "the pinched tile", strip_lengths[0], self.minimum_strip_tiles
                    )
                self.linger()
        finally:
            self.model_directory.cleanup()
        return Pr2StripPeelingOutcome(placement_error, strip_lengths[0])


def main(
    avocado_directory: Path | None = None, headless: bool = False
) -> Pr2StripPeelingOutcome:
    """
    Run the demonstration and report what the PR2 achieved.

    :param avocado_directory: The avocado model's folder; defaults to the folder named
        by :attr:`EnvironmentVariable.AVOCADO_DIRECTORY`, or else the avocado that comes
        with CRAM.
    :param headless: Whether the physics runs without a viewer.
    :return: What the PR2 achieved.
    """
    outcome = Pr2AvocadoPeelingDemonstration(
        avocado_folder(avocado_directory), headless=headless
    ).run()
    print(
        f"The PR2 laid the avocado {outcome.placement_error * 1000:.0f} mm from the "
        f"board's centre and peeled a strip of {outcome.strip_length} tiles, which it "
        f"laid down beside the board."
    )
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The PR2 puts an avocado on a cutting board and peels a strip of it."
    )
    parser.add_argument(
        "avocado_directory",
        type=Path,
        nargs="?",
        help="the avocado model's folder; by default the one that comes with CRAM",
    )
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    arguments = parser.parse_args()
    main(arguments.avocado_directory, arguments.headless)
