from __future__ import annotations
from dataclasses import dataclass
from typing import Union, List, Tuple

import numpy as np

from rl_greenhouse.greenhouse.types.base import BaseDataClass
from rl_greenhouse.greenhouse.types.plant import Plant
from rl_greenhouse.greenhouse.types.state import State
from rl_greenhouse.greenhouse.types.weather import Weather
from rl_greenhouse.greenhouse.types.info import Info


@dataclass
class Observation(BaseDataClass):
    """
    Describes the entire greenhouse state at a particular moment in time.
    """

    info: Info
    plant: Plant
    state: State
    weather: Weather

    LABELS = ["info", "plant", "state", "weather"]
    FLAT_LABELS = Info.LABELS + Plant.LABELS + State.LABELS + Weather.LABELS
    INTERVALS = {**Info.INTERVALS, **Plant.INTERVALS, **State.INTERVALS, **Weather.INTERVALS}

    @classmethod
    def get_default(cls) -> Observation:
        return Observation(Info(), Plant(), State(), Weather())

    def get(self, attribute: str) -> float:
        if attribute in Info.LABELS:
            return getattr(self.info, attribute)
        if attribute in Plant.LABELS:
            return getattr(self.plant, attribute)
        if attribute in State.LABELS:
            return getattr(self.state, attribute)
        if attribute in Weather.LABELS:
            return getattr(self.weather, attribute)

    def to_numpy(self) -> np.array:
        return np.array(
            [
                *list(self.info.to_dict().values()),
                *list(self.plant.to_dict().values()),
                *list(self.state.to_dict().values()),
                *list(self.weather.to_dict().values()),
            ],
            dtype=np.float32,
        )

    def to_numpy_2d(self) -> np.array:
        """Returns a numpy array of this class"""
        return np.expand_dims(self.to_numpy(), axis=0)

    def normalize(self) -> Observation:
        values = {
            "info": self.info.normalize(),
            "plant": self.plant.normalize(),
            "state": self.state.normalize(),
            "weather": self.weather.normalize(),
        }
        return self.__class__(**values)

    def denormalize(self) -> Observation:
        values = {
            "info": self.info.denormalize(),
            "plant": self.plant.denormalize(),
            "state": self.state.denormalize(),
            "weather": self.weather.denormalize(),
        }
        return self.__class__(**values)

    def unpack(self) -> (Info, Plant, State, Weather):
        return self.info, self.plant, self.state, self.weather

    @classmethod
    def from_n_values(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], np.ndarray):
                args = args[0].tolist()
        else:
            args = list(args)
        return Observation(
            Info(*[args.pop(0) for _ in range(len(Info.LABELS))]),
            Plant(*[args.pop(0) for _ in range(len(Plant.LABELS))]),
            State(*[args.pop(0) for _ in range(len(State.LABELS))]),
            Weather(*[args.pop(0) for _ in range(len(Weather.LABELS))]),
        )


def pack_observations(
    info: Union[List[Info], Info],
    plants: Union[List[Plant], Plant],
    states: Union[List[State], State],
    weather: Union[List[Weather], Weather],
) -> Union[List[Observation], Observation]:
    """
    Packs iterables or non iterables into Observation iterables or non iterables.

    :param info:
    :param states:
    :param plants:
    :param weather:
    """
    if isinstance(info, list):
        return [Observation(*tuple_) for tuple_ in zip(info, plants, states, weather)]
    return Observation(info, plants, states, weather)


def unpack_observations(
    observations: Union[List[Observation], Observation]
) -> Tuple[
    Union[List[Info], Info],
    Union[List[Plant], Plant],
    Union[List[State], State],
    Union[List[Weather], Weather],
]:
    """
    Unpacks iterable or non-iterable full state into its components.
    :rtype: object
    :param observations:
    :return:
    """
    return (
        [full_state.info for full_state in observations],
        [full_state.plant for full_state in observations],
        [full_state.state for full_state in observations],
        [full_state.weather for full_state in observations],
    )


if __name__ == "__main__":
    print(f"({len(Observation(Info(), Plant(), State(), Weather()).FLAT_LABELS)},)")
    print(Observation(Info(), Plant(), State(), Weather()).FLAT_LABELS)
