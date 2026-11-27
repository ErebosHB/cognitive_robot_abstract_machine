"""
What the parts of a fruit are made of.
"""

from __future__ import annotations

from enum import StrEnum

# %% materials


class FruitMaterial(StrEnum):
    """
    What a part of a fruit is made of; the value ends the name of the part's geom.
    """

    SKIN = "skin"
    FLESH = "flesh"
    PIT = "pit"
