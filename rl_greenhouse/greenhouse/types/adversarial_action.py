from dataclasses import dataclass

from rl_greenhouse.greenhouse.types.base import BaseDataClass


@dataclass
class AdversarialAction(BaseDataClass):
    """
    The default action interface.

    :param heat_offset: The amount of heat injected / removed in W/m²
    :param light_offset: The amount of light added / removed in PAR Light
    :param carbon_offset: The amount of carbon added / removed in kgCO2/s/m²
    :param humidity_absolute_offset: The amount of evaporated water added / removed in kg/s/m²
    """

    heat_offset: float = 0
    light_offset: float = 0
    carbon_offset: float = 0
    humidity_absolute_offset: float = 0

    INTERVALS = {
        "heat_offset": (-10, 10),  # W/m² - 5% of heating power
        "light_offset": (-25, 25),  # Less than lamps - Approx 5% of illumination peak (500)
        "carbon_offset": (-0.5e-08, 0.5e-08),  # 100x smaller than injection action
        "humidity_absolute_offset": (-0.0000005, 0.0000005),  # kg/s/m²
    }

    LABELS = list(INTERVALS.keys())


if __name__ == "__main__":
    import numpy as np

    action = AdversarialAction()
    np_action = action.to_numpy_2d()
    print(action)
    print(np_action)
    print(action.normalize().denormalize())
    print("from numpy")
    print(AdversarialAction.from_numpy(np.array([0, 1, 2, 3])))

    some_action = AdversarialAction(5, -25, 0, 0)
    print("Some Action")
    print(some_action)
    print("Doubled")
    print(some_action.scale(2))
    print("Halved")
    print(some_action.scale(0.5))
