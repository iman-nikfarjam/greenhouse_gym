from rl_greenhouse.greenhouse.types import State, Weather, Plant, Observation, Info
from rl_greenhouse.greenhouse.components.base import PlantComponent


class ReplayPlant(PlantComponent):
    """
    A lettuce plant replay
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters, parameters_used={"plants_to_replay": "The plant states to replay"}
        )
        plants = hyper_parameters["plants_to_replay"]
        self.sell_overgrown_lettuce_for_max_price = False

        if isinstance(plants[0], Observation):
            plants = [full_state.plant for full_state in plants]

        if not isinstance(plants[0], Plant):
            raise TypeError(f"Plants of type [{type(plants[0])}] are not supported.")

        self.plants = plants

    def step(self, info: Info, plant: Plant, state: State, weather: Weather, delta_time: float) -> (Plant, State, dict):
        """
        Calculates the resulting plant state from the current state in the greenhouse.

        :param info: The current extra information state
        :param plant: The current state of the plant
        :param state: The current state of the system
        :param weather: The weather at this time
        :param delta_time: Delta time in seconds
        """
        idx = round(info.time) - 1
        return self.plants[idx], state, {}

    def current_selling_price(self, plant: Plant) -> float:
        """Returns the current selling price in € / plant"""
        dist_from_ideal_weight = abs(plant.mass - 250)
        if dist_from_ideal_weight > 40:
            return 0
        elif dist_from_ideal_weight < 20:
            return (0.5 - 0.1 * dist_from_ideal_weight / 20) * plant.quality
        return (0.4 - 0.4 * ((dist_from_ideal_weight - 20) / 20)) * plant.quality

    def reset(self) -> Plant:
        """
        Reset this subsystem.
        """
        return self.plants[0]
