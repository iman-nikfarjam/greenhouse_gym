from rl_greenhouse.greenhouse.types import State, Weather, Plant, Info
from rl_greenhouse.greenhouse.components.base import PlantComponent


class NoPlant(PlantComponent):
    """
    Not a single plant in the greenhouse.
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(hyper_parameters=hyper_parameters, parameters_used={})

    def step(self, info: Info, plant: Plant, state: State, weather: Weather, delta_time: float) -> (Plant, State, dict):
        """
        Calculates the resulting plant state from the current state in the greenhouse.

        :param info: The current extra information state
        :param plant: The current state of the plant
        :param state: The current state of the system
        :param weather: The weather at this time
        :param delta_time: Delta time in seconds
        """
        return plant, state, {}

    def current_selling_price(self, plant: Plant) -> float:
        """
        Calculates the current selling price of the plant
        :param plant: The current state of the plant
        """
        return 0

    def reset(self) -> Plant:
        """
        Reset this subsystem.
        """
        return Plant(
            mass=1,
            quality=1,
        )
