from dataclasses import dataclass

from rl_greenhouse.greenhouse.types.base import BaseDataClass


@dataclass
class Action(BaseDataClass):
    """
    The default action interface.

    heater_power: The power for the heating system in [W/m²]
    ventilation_position: The setpoint for the ventilation where 0 is closed and 1 is open
    co2_flow_rate: Flow rate of CO2 into the greenhouse in [kg/m²s]
    screen_blackout_position: The blackout screen position where 0 is closed and 1 is open
    screen_transparent_position: The transparent screen position where 0 is closed and 1 is open
    enable_lamps: Whether the lamps are on or off
    """

    heater_power: float = 0
    ventilation_position: float = 0
    co2_flow_rate: float = 0
    screen_blackout_position: float = 0
    screen_transparent_position: float = 0
    enable_lamps: float = 0

    INTERVALS = {
        "heater_power": (0, 200),
        "ventilation_position": (0, 1),
        "co2_flow_rate": (0, 0.5e-06),
        "screen_blackout_position": (0, 1),
        "screen_transparent_position": (0, 1),
        "enable_lamps": (0, 1),
    }

    LABELS = list(INTERVALS.keys())


if __name__ == "__main__":
    import numpy as np

    action = Action()
    np_action = action.to_numpy_2d()
    print(action)
    print(np_action)
    print(action.normalize().denormalize())
    print("from numpy")
    print(Action.from_numpy(np.array([0, 1, 2, 3, 4, 5])))
