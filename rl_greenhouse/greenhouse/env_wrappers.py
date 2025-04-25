import sys
import math
from typing import Union

import gym
import numpy as np
from gym.spaces import Box, Dict
from ray.rllib import MultiAgentEnv

from rl_greenhouse.greenhouse import env as greenhouse_gym
from rl_greenhouse.greenhouse.types import Observation, Action, AdversarialAction
from rl_greenhouse.greenhouse.env import GreenhouseNumpy, Greenhouse, DEFAULT_GREENHOUSE_CONFIG


class WrappedGreenhouse(gym.Wrapper):
    """
    Stacks wrappers on top of each other to create a gym environment with multiple sequentially applied wrappers.
    Returns a functioning environment with multi wrapper layers.

    WRAPTASTIC!

    Be like a union.

    Add this to the environment config:

        "wrappers": A list of wrapper types, where the leftmost wrapper is applied first and the rightmost last.

        "base_env": The base env to generate the stacked wrapper.

    :param config: Env config as defined in RLLIB.
    """
    name = "wrapper_greenhouse"

    def __init__(self, config: dict):
        wrappers = config["wrappers"]
        if wrappers is None:
            super().__init__(GreenhouseNumpy(config))
        else:
            BaseEnvType = wrappers[0] if isinstance(wrappers[0], Greenhouse) else getattr(greenhouse_gym, wrappers[0])
            env = BaseEnvType(config)

            for next_wrap in wrappers[1:]:
                NextWrapperType = next_wrap \
                    if isinstance(next_wrap, (Greenhouse, gym.Wrapper)) \
                    else getattr(sys.modules[__name__], next_wrap)
                env = NextWrapperType(config, env=env)

            super().__init__(env)


class WrappedGreenhouseMultiAgent(WrappedGreenhouse, MultiAgentEnv):
    """
    MultiAgent Mixin class that combines the WrappedGreenhouse interface with the required interface for RLLIB.
    """


class StackedFrameWrapper(gym.Wrapper):
    """
    Wrapper used to stack n frames.

    Initially, the first observation is a concatenation n first frames.
    """

    def __init__(self, config: dict, env: Union[GreenhouseNumpy, gym.Wrapper]):
        super().__init__(env)
        self.env = env
        self.nr_frames_stacked = config["nr_frames_stacked"]
        assert self.nr_frames_stacked >= 1, "nr_frames_stacked cannot be less than 1"

        self.observation_space = Box(
            low=np.hstack([self.env.observation_space.low] * self.nr_frames_stacked),
            high=np.hstack([self.env.observation_space.high] * self.nr_frames_stacked),
            dtype=self.env.observation_space.dtype,
        )
        self.buffer = np.zeros(
            (self.nr_frames_stacked, self.env.observation_space.shape[0]),
            dtype=self.env.observation_space.dtype
        )

    def reset(self):
        observation = self.env.reset()
        for i in range(self.nr_frames_stacked):
            self.buffer[i, :] = observation
        return self.buffer.flatten()

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1, ...] = observation
        return self.buffer.flatten(), reward, done, info


class FeatureEngineeringWrapper(gym.ObservationWrapper):
    """

    Assumes:
        - Input is of size Observation.to_numpy()
        - Input is normalized

    Removes:
        - Cumulative heat / carbon / electricity usage and other cost related variables contained in TO_DELETE_COLUMNS

    Applies:
        - Cyclical transformations on week, hour
    """
    CYCLIC_COLUMNS = ["week", "hour"]

    TO_DELETE_COLUMNS = [
        "time",
        "cum_electricity_usage", "cum_heat_usage", "cum_carbon_usage",
        "price_heat", "price_carbon",
        "costs_var_electricity", "costs_var_heat", "costs_var_carbon",
        "average_head_per_m2", "gains_plant", "costs_var_total", "costs_fix_total", "balance",
    ]

    def __init__(self, config: dict, env: Union[GreenhouseNumpy, gym.Wrapper]):
        super().__init__(env)
        assert env.observation_space == GreenhouseNumpy.observation_space, \
            f"Obs space of {env.observation_space} violates assumption of this wrapper."

        self.cyclical_indices = [Observation.FLAT_LABELS.index(col) for col in self.CYCLIC_COLUMNS]
        self.indices_to_delete = \
            [Observation.FLAT_LABELS.index(col) for col in self.TO_DELETE_COLUMNS + self.CYCLIC_COLUMNS]

        resulting_space = env.observation_space.shape[0] + len(self.CYCLIC_COLUMNS) - len(self.TO_DELETE_COLUMNS)
        self.observation_space = Box(low=0, high=1, shape=(resulting_space,), dtype=np.float32)

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        unprocessed_cyclical_features = [observation[idx] for idx in self.cyclical_indices]

        cyclical_features = []
        for feat in unprocessed_cyclical_features:
            # transformed to the 0, 1 domain (also with half amplitude)
            cyclical_features.append(0.5 + 0.5 * math.cos(2 * math.pi * feat))
            cyclical_features.append(0.5 + 0.5 * math.sin(2 * math.pi * feat))

        observation = np.delete(observation, self.indices_to_delete)
        observation = np.concatenate([observation, np.array(cyclical_features)], dtype=np.float32)
        return observation


class GaussianNoiseWrapper(gym.ObservationWrapper):
    """
    Gaussian Noise domain randomization wrapper. Adds noise with mu / std. Can be used to introduce variance or bias
    in the observation.

    Assumes:
        - Observation space is of type Box with dtype float

    Applies:
        - Random noise with noise ~ N(mu, std), clips the result to the original observation space.
    """

    def __init__(self, config: dict, env: Union[GreenhouseNumpy, gym.Wrapper]):
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "Observation Space should be of type Box."
        self.gaussian_noise_mu = config.get('gaussian_noise_mu', DEFAULT_GREENHOUSE_CONFIG['gaussian_noise_mu'])
        self.gaussian_noise_std = config.get('gaussian_noise_std', DEFAULT_GREENHOUSE_CONFIG['gaussian_noise_std'])
        self.observation_space = env.observation_space

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        standard_normal_samples = np.random.standard_normal(size=self.observation_space.shape)
        normal_samples = standard_normal_samples * self.gaussian_noise_std + self.gaussian_noise_mu
        return np.clip(observation + normal_samples, self.observation_space.low, self.observation_space.high)


class AdversarialWrapper(gym.Wrapper):
    """
    Adversarial Agent Wrapper that allows a protagonist and an adversarial to compete in a single environment.
    Steps happen simultaneously where each agent uses its own action space.

    Requires the MultiAgent version of the WrappedGreenhouse class.
    """
    PROT = "protagonist"
    ADV = "adversarial"

    ACTION_MAPPING = {PROT: "action", ADV: "adversarial_action"}

    def __init__(self, config: dict, env: Union[GreenhouseNumpy, gym.Wrapper]):
        super().__init__(env)
        self.env = env
        self.observation_spaces = {self.PROT: self.env.observation_space, self.ADV: self.env.observation_space}
        self.action_spaces = {
            self.PROT: Box(low=0, high=1, shape=(len(Action.LABELS),), dtype=np.float32),
            self.ADV: Box(low=0, high=1, shape=(len(AdversarialAction.LABELS),), dtype=np.float32),
        }
        self.observation_space = Dict(**self.observation_spaces)
        self.action_space = Dict(**self.action_spaces)

    def reset(self):
        observation = self.env.reset()
        return {self.PROT: observation, self.ADV: observation}

    def step(self, actions):
        input_dict = {self.ACTION_MAPPING[agent_name]: action for agent_name, action in actions.items()}
        observation, reward, done, info = self.env.step(**input_dict)

        return {self.PROT: observation, self.ADV: observation}, \
               {self.PROT: reward, self.ADV: -reward}, \
               {self.PROT: done, self.ADV: done, "__all__": done}, \
               {self.PROT: info, self.ADV: info}
