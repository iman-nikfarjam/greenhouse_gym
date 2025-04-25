import copy

from rl_greenhouse.greenhouse.types import Observation, Info
from rl_greenhouse.greenhouse.types import State, Weather, Action
from rl_greenhouse.greenhouse.components.base import GreenhouseComponent


class ReplayState(GreenhouseComponent):
    """
    Replays given states to the environment by imputing the given states into the simulator at each step.
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters,
            parameters_used={
                "states_to_replay": "List of states that will be replayed to the system.",
                "variable_blacklist": "List of variables to exclude from this replay function.",
            },
        )
        states = hyper_parameters["states_to_replay"]

        if isinstance(states[0], Observation):
            states = [full_state.state for full_state in states]

        if not isinstance(states[0], State):
            raise TypeError(f"State of type [{type(states[0])}] is not supported.")

        self.states = states
        self.variable_blacklist = hyper_parameters.get("variable_blacklist", [])

    def step(self, info: Info, state: State, weather: Weather, action: Action, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param delta_time: Delta time in seconds
        """
        idx = round(info.time)
        modified_state = copy.deepcopy(self.states[idx])
        return self._unmodify_states_in_blacklist(state, modified_state)

    def _unmodify_states_in_blacklist(self, unmodified_state: State, modified_state: State) -> State:
        for variable in self.variable_blacklist:
            modified_state.__setattr__(variable, unmodified_state.__getattribute__(variable))
        return modified_state
