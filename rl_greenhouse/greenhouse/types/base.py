from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BaseDataClass:
    """
    Standard types class
    """

    LABELS = []
    INTERVALS = {}

    @classmethod
    def from_numpy(cls, numpy_array: np.array) -> BaseDataClass:
        return cls(*numpy_array.reshape(-1))

    def to_dict(self) -> Dict[str, float]:
        return {label: self.__dict__[label] for label in self.LABELS}

    def to_numpy(self) -> np.array:
        """Returns a numpy array of this class"""
        return np.array(list(self.to_dict().values()), dtype=np.float32)

    def to_numpy_2d(self) -> np.array:
        """Returns a numpy array of this class"""
        return np.expand_dims(self.to_numpy(), axis=0)

    def __len__(self):
        return len(self.LABELS)

    def __str__(self):
        return "\n".join([f"{field}: {value}" for field, value in self.to_dict().items()])

    def normalize(self) -> BaseDataClass:
        values = {
            label: (self.__getattribute__(label) - self.INTERVALS[label][0])
            / (self.INTERVALS[label][1] - self.INTERVALS[label][0])
            for label in self.LABELS
        }
        return self.__class__(**values)

    def denormalize(self) -> BaseDataClass:
        values = {
            label: (self.__getattribute__(label) * (self.INTERVALS[label][1] - self.INTERVALS[label][0]))
            + self.INTERVALS[label][0]
            for label in self.LABELS
        }
        return self.__class__(**values)

    def scale(self, scale: float) -> BaseDataClass:
        values = {label: self.__getattribute__(label) * scale for label in self.LABELS}
        return self.__class__(**values)


if __name__ == "__main__":
    data = BaseDataClass()
    np_data = data.to_numpy_2d()
    print(data)
    print(np_data)
