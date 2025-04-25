"""


"""
from dataclasses import dataclass

from rl_greenhouse.greenhouse.types.base import BaseDataClass


@dataclass
class Plant(BaseDataClass):
    """
    The default plant interface.

    mass: The weight of the plants in grams
    quality: Fraction of dry matter from 0 to 1
    """

    mass: float = 2.26
    quality: float = 1
    plants_per_m2: float = 25

    INTERVALS = {
        "mass": (0, 400),
        "quality": (0, 1),
        "plants_per_m2": (1, 80),
    }
    LABELS = list(INTERVALS.keys())


if __name__ == "__main__":
    plant_state = Plant()
    np_plant_state = plant_state.to_numpy_2d()
    norm_plant_state = plant_state.normalize()
    denormalized_plant_state = norm_plant_state.denormalize()
    print(plant_state)
    print(np_plant_state)
    print(norm_plant_state)
    print(denormalized_plant_state)
