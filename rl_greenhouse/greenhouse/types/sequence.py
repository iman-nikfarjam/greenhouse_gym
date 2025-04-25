from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd

from rl_greenhouse.greenhouse.types.state import State
from rl_greenhouse.greenhouse.types.plant import Plant
from rl_greenhouse.greenhouse.types.weather import Weather
from rl_greenhouse.greenhouse.types.action import Action
from rl_greenhouse.greenhouse.types.observation import Observation
from rl_greenhouse.greenhouse.types.info import Info
from rl_greenhouse.greenhouse.types.observation import unpack_observations, pack_observations


class Sequence:
    """
    Creates a sequence from starting state, full states and actions.

    Full States must be one longer than actions.
    The reason is you need a starting state and ending state. Therefore you need one action less.

    Example state transitions:
    s0 -> a0 -> s1 -> a1 -> s2

    :param observations: All full states from f(0) to f(t)
    :param actions: All actions from a(0) to a(t-1)
    """

    NP_ARRAY_LABELS = Observation.FLAT_LABELS + Action.LABELS

    def __init__(
            self,
            observations: List[Observation],
            actions: List[Action],
            env_infos: Optional[List[dict]] = None,
    ):
        assert len(observations) == len(actions) + 1, (
            "Full States must be one longer than actions.\n"
            "The reason is you need a starting state and ending state. Therefore you need one action less.\n"
            "s0 -> a0 -> s1 -> a1 -> s2"
        )
        self.observations = observations
        infos, plants, states, weathers = unpack_observations(observations)
        self.infos: List[Info] = infos
        self.plants: List[Plant] = plants
        self.states: List[State] = states
        self.weathers: List[Weather] = weathers
        self.actions: List[Action] = actions
        self.env_infos: Optional[List[dict]] = env_infos

    def normalize(self) -> Sequence:
        return Sequence(
            observations=pack_observations(
                [info.normalize() for info in self.infos],
                [plant.normalize() for plant in self.plants],
                [state.normalize() for state in self.states],
                [weather.normalize() for weather in self.weathers],
            ),
            actions=[action.normalize() for action in self.actions],
            env_infos=self.env_infos,
        )

    def denormalize(self) -> Sequence:
        return Sequence(
            observations=pack_observations(
                [info.denormalize() for info in self.infos],
                [plant.denormalize() for plant in self.plants],
                [state.denormalize() for state in self.states],
                [weather.denormalize() for weather in self.weathers],
            ),
            actions=[action.denormalize() for action in self.actions],
            env_infos=self.env_infos,
        )

    def to_numpy(self, normalized: bool, variables: List[str] = None) -> np.array:
        """
        Creates a numpy array from the observation. Leaves out the final obs. One row in the array contains s0 and a0
        :param normalized: Whether to normalize the output.
        :param variables: Variable whitelist. All other columns are excluded.
        """
        array = self._to_numpy_unfiltered(normalized)
        if variables is not None:
            whitelist = [self.labels().index(variable) for variable in self.labels(variables)]
            array = array[:, whitelist]
        return array

    def labels(self, variables: List[str] = None) -> List[str]:
        if variables is not None:
            return variables
        return self.NP_ARRAY_LABELS

    def __iter__(self):
        return iter([self.infos, self.plants, self.states, self.weathers, self.actions])

    def __len__(self):
        return len(self.observations)

    def __eq__(self, other: Sequence):
        if not isinstance(other, Sequence):
            return False
        if len(self) != len(other):
            return False
        for obs_a, obs_b in zip(self.observations, other.observations):
            if not np.all(obs_a.to_numpy() == obs_b.to_numpy()):
                return False
        for action_a, action_b in zip(self.actions, other.actions):
            if not np.all(action_a.to_numpy() == action_b.to_numpy()):
                return False
        return True

    def _get_start_stop(self, subscript) -> (int, int):
        if subscript.start is None:
            start = 0
        else:
            start = subscript.start
        if subscript.stop is None:
            stop = len(self)
        else:
            stop = subscript.stop
        return start, stop

    def _new_sequence_by_slice(self, subscript) -> Sequence:
        start, stop = self._get_start_stop(subscript)
        return Sequence(
            self.observations[start:stop],
            self.actions[start: stop - 1],  # s[1:3] for s0, s1, s2, a0, a1 -> s1, s2, a1
            self.env_infos[start: stop - 1] if self.env_infos else None,
        )

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return self._new_sequence_by_slice(subscript)
        else:
            # Do your handling for a plain index
            raise NotImplementedError()

    def _create_obs_and_action_array(self, normalized: bool) -> (np.array, np.array):
        if normalized:
            observations = [full_state.normalize().to_numpy_2d() for full_state in self.observations]
            actions = [action.normalize().to_numpy_2d() for action in self.actions]
            actions.append(np.ones_like(actions[0]) * float('nan'))
        else:
            observations = [full_state.to_numpy_2d() for full_state in self.observations]
            actions = [action.to_numpy_2d() for action in self.actions]
            actions.append(np.ones_like(actions[0]) * float('nan'))

        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        return observations, actions

    def _to_numpy_unfiltered(self, normalized: bool) -> np.array:
        observations, actions = self._create_obs_and_action_array(normalized)
        array = np.concatenate((observations, actions), axis=1)
        return array

    def to_pandas(self, normalized: bool) -> pd.DataFrame:
        array = self._to_numpy_unfiltered(normalized)
        return pd.DataFrame(columns=self.NP_ARRAY_LABELS, data=array)


class SequenceExtraInfo(Sequence):
    """
    Sequence with extra information. Extra information can be converted to Numpy arrays.
    """

    def __init__(
            self,
            observations: List[Observation],
            actions: List[Action],
            extra_information: List[List[float]],
            extra_labels: List[str],
            env_infos: Optional[List[dict]] = None,
    ):
        # if len(extra_information) > 0:
        #     assert len(extra_information[0]) == len(
        #         extra_labels
        #     ), f"Provided {len(extra_information[0])} extra label columns but only {len(extra_labels)} extra labels."
        # assert len(observations) == len(extra_information), (
        #     f"Extra information must be of same length as the observations, "
        #     f"not {len(extra_information)} and {len(observations)}"
        # )
        super().__init__(observations, actions, env_infos=env_infos)
        self.extra_information = extra_information
        self.extra_labels = extra_labels

    def normalize(self) -> Sequence:
        raise NotImplementedError("Not implemented for extra info sequence")

    def denormalize(self) -> Sequence:
        raise NotImplementedError("Not implemented for extra info sequence")

    def labels(self, variables: List[str] = None) -> List[str]:
        if variables is not None:
            return variables + self.extra_labels
        return self.NP_ARRAY_LABELS + self.extra_labels

    def __iter__(self):
        return iter(
            [self.infos, self.plants, self.states, self.weathers, self.actions, self.extra_information])

    def _new_sequence_by_slice(self, subscript) -> Sequence:
        start, stop = self._get_start_stop(subscript)
        return SequenceExtraInfo(
            # s[1:3] for s0, s1, s2, a0, a1 -> s1, s2, a1
            self.observations[start:stop],
            self.actions[start: stop - 1],
            self.extra_information[start:stop],
            self.extra_labels,
            env_infos=self.env_infos[start: stop - 1],
        )

    def _to_numpy_unfiltered(self, normalized: bool) -> np.array:
        observations, actions = self._create_obs_and_action_array(normalized)
        extra_information = np.array(self.extra_information).reshape(-1, len(self.extra_labels))
        array = np.concatenate((observations, actions, extra_information), axis=1)
        return array
