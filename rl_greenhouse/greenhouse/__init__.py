from typing import Union

from rl_greenhouse.greenhouse.config import DEFAULT_GREENHOUSE_CONFIG
from rl_greenhouse.greenhouse.env import GreenhouseNumpy, Greenhouse
from rl_greenhouse.greenhouse.env_wrappers import WrappedGreenhouse, WrappedGreenhouseMultiAgent

GREENHOUSE_TYPE = Union[GreenhouseNumpy, WrappedGreenhouse]
