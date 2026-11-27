"""
The four demonstrations with the solid avocado, to be chosen by name.
"""

from __future__ import annotations

from enum import StrEnum

from cutting_demo import AvocadoCuttingDemonstration
from scooping_demo import AvocadoScoopingDemonstration
from skin_peeling_demo import SkinPeelingDemonstration
from slicing_demo import AvocadoSlicingDemonstration
from solid_avocado_demonstration import SolidAvocadoDemonstration

# %% the demonstrations


class Demonstration(StrEnum):
    """
    The demonstrations with the solid avocado.
    """

    CUTTING = "cutting"
    PEELING = "peeling"
    SLICING = "slicing"
    SCOOPING = "scooping"

    def create(
        self, *, headless: bool, playback_speed: float = 1.0, pit_seed: int | None
    ) -> SolidAvocadoDemonstration:
        """
        :param headless: Whether to run without a viewer.
        :param playback_speed: Simulated seconds shown per second in the viewer.
        :param pit_seed: The seed of the avocado's random pit; None draws one.
        :return: The demonstration, set up but not run yet.
        """
        match self:
            case Demonstration.CUTTING:
                demonstration_type = AvocadoCuttingDemonstration
            case Demonstration.PEELING:
                demonstration_type = SkinPeelingDemonstration
            case Demonstration.SLICING:
                demonstration_type = AvocadoSlicingDemonstration
            case Demonstration.SCOOPING:
                demonstration_type = AvocadoScoopingDemonstration
        return demonstration_type(
            headless=headless, playback_speed=playback_speed, pit_seed=pit_seed
        )
