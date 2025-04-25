"""


"""
from dataclasses import dataclass

from rl_greenhouse.greenhouse.types.base import BaseDataClass


@dataclass
class State(BaseDataClass):
    """
    The default observation interface.

    temperature: The temperature in the greenhouse in degrees Celsius
    par_light: The par light in µ mol / s just above the plant
    carbon_ppm: The carbon dioxide ppm in the greenhouse
    """

    temperature_air: float = 16
    heating_pipe_temperature: float = 0
    relative_humidity: float = 0.8
    par_light: float = 0
    carbon_ppm: float = 400
    wind_speed: float = 0
    electric_power: float = 0  # In W/m²

    INTERVALS = {
        "temperature_air": (-20, 60),
        "heating_pipe_temperature": (0, 100),
        "relative_humidity": (0, 1),
        "par_light": (0, 2000),
        "carbon_ppm": (0, 6000),
        "wind_speed": (0, 20),
        "electric_power": [0, 1000],
    }
    LABELS = list(INTERVALS.keys())


if __name__ == "__main__":
    state = State()
    np_state = state.to_numpy_2d()
    print(state)
    print(np_state)
    print(state.normalize().denormalize())
