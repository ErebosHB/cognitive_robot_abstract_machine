"""
Runs a demonstration with the solid avocado headless and writes the pieces it ends with
to a USD file that Isaac Sim can open and simulate, on a floor at the height of the
cutting board.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from demonstrations import Demonstration
from fruit_stage import FruitStage

# %% entry point


def main(demonstration: Demonstration, output: Path, pit_seed: int | None) -> None:
    """
    Run a demonstration and write the pieces it ends with to a USD file.

    :param demonstration: The demonstration.
    :param output: The USD file; its extension, such as ``.usda`` or ``.usd``, chooses
        the format.
    :param pit_seed: The seed of the avocado's random pit; None draws one.
    """
    finished = demonstration.create(headless=True, pit_seed=pit_seed)
    finished.run()
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
