"""
Runs a demonstration with the solid avocado headless and writes the pieces it ends with
to a USD file that Isaac Sim can open and simulate, on a floor at the height of the
cutting board.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path

from cutting_demo import AvocadoCuttingDemonstration
from fruit_stage import FruitStage
from scooping_demo import AvocadoScoopingDemonstration
from skin_peeling_demo import SkinPeelingDemonstration
from slicing_demo import AvocadoSlicingDemonstration
from solid_avocado_demonstration import SolidAvocadoDemonstration

# %% the demonstrations


class Demonstration(StrEnum):
    """
    The demonstrations whose pieces can be written.
    """

    CUTTING = "cutting"
    PEELING = "peeling"
    SLICING = "slicing"
    SCOOPING = "scooping"

    def run(self, pit_seed: int | None) -> SolidAvocadoDemonstration:
        """
        :param pit_seed: The seed of the avocado's random pit; None draws one.
        :return: The demonstration, run headless to its end.
        """
        match self:
            case Demonstration.CUTTING:
                demonstration = AvocadoCuttingDemonstration(
                    headless=True, pit_seed=pit_seed
                )
            case Demonstration.PEELING:
                demonstration = SkinPeelingDemonstration(
                    headless=True, pit_seed=pit_seed
                )
            case Demonstration.SLICING:
                demonstration = AvocadoSlicingDemonstration(
                    headless=True, pit_seed=pit_seed
                )
            case Demonstration.SCOOPING:
                demonstration = AvocadoScoopingDemonstration(
                    headless=True, pit_seed=pit_seed
                )
        demonstration.run()
        return demonstration


# %% entry point


def main(demonstration: Demonstration, output: Path, pit_seed: int | None) -> None:
    """
    Run a demonstration and write the pieces it ends with to a USD file.

    :param demonstration: The demonstration.
    :param output: The USD file; its extension, such as ``.usda`` or ``.usd``, chooses
        the format.
    :param pit_seed: The seed of the avocado's random pit; None draws one.
    """
    finished = demonstration.run(pit_seed)
    stage = FruitStage(finished.fruit_world, floor_height=finished.board_size[2])
    stage.write(output)
    print(
        f"Wrote the {len(finished.fruit.pieces)} pieces the {demonstration} "
        f"demonstration ended with, its pit from seed {finished.pit_seed}, to {output}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write the pieces of a solid avocado demonstration to USD."
    )
    parser.add_argument(
        "demonstration",
        type=Demonstration,
        choices=list(Demonstration),
        help="the demonstration to run",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="the USD file to write"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="the seed of the random pit"
    )
    arguments = parser.parse_args()
    output = arguments.output or Path(f"avocado_{arguments.demonstration}.usda")
    main(arguments.demonstration, output, arguments.seed)
