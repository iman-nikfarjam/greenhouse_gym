import unittest
from typing import List

import numpy as np

from rl_greenhouse.greenhouse.types import Action, Observation
from rl_greenhouse.greenhouse.env import GreenhouseNumpy
from rl_greenhouse.greenhouse.components.economic import BasicCostModel
from rl_greenhouse.greenhouse.components.greenhouse import Co2Supply, StandardLights, WallConduction, Ventilation, Heater
from rl_greenhouse.greenhouse.components.plant import Lettuce
from rl_greenhouse.greenhouse.components.weather import HoekVanHollandWeatherStation


def play_actions_on_numpy_env(env: GreenhouseNumpy, all_actions: List[Action], stop_when_done=False) -> \
        (List[np.array], List[np.array]):
    """
    Plays the actions on the given environment, returns a sequence.

    :param env:
    :param all_actions:
    :param stop_when_done:
    """
    time = 0
    observations = [env.reset()]
    for action in all_actions:
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        time += 1
        if done and stop_when_done:
            return observations
    return observations, all_actions


class GreenhouseTests(unittest.TestCase):
    """
    To test:

    """

    CONSTANT_ACTION = constant_action = Action(
        heater_power=23,
        co2_flow_rate=0.5e-6,
        enable_lamps=0,
        ventilation_position=0,
        screen_transparent_position=0.6,
    ).normalize().to_numpy()

    CONSTANT_ACTIONS_10_DAYS = [CONSTANT_ACTION] * 10 * 24

    def test_full_system(self):
        env_config = {
            "greenhouse_model_components": [Heater, WallConduction, StandardLights, Ventilation, Co2Supply],
            "weather_model": HoekVanHollandWeatherStation,
            "plant_model": Lettuce,
            'economic_model': BasicCostModel,
            'growing_cycle': 10,
        }
        greenhouse = GreenhouseNumpy(env_config)
        play_actions_on_numpy_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_input_and_output_values(self):
        env_config = {
            "greenhouse_model_components": [Heater, WallConduction, StandardLights, Ventilation, Co2Supply],
            "weather_model": HoekVanHollandWeatherStation,
            "plant_model": Lettuce,
            'economic_model': BasicCostModel,
            'growing_cycle': 10,
        }
        greenhouse = GreenhouseNumpy(env_config)
        all_obs, all_actions = play_actions_on_numpy_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

        for obs in all_obs:
            self.assertTrue(np.all(obs >= 0), f"All observation values must be at least 0, not {np.min(obs)}")
            self.assertTrue(np.all(obs <= 1), f"All observation values must be at most 1, not {np.max(obs)}")

        for action in all_actions:
            self.assertTrue(np.all(action >= 0), f"All action values must be at least 0, not {np.min(action)}")
            self.assertTrue(np.all(action <= 1), f"All action values must be at most 1, not {np.max(action)}")

    def test_input_and_output_shape(self):
        env_config = {
            "greenhouse_model_components": [Heater, WallConduction, StandardLights, Ventilation, Co2Supply],
            "weather_model": HoekVanHollandWeatherStation,
            "plant_model": Lettuce,
            'economic_model': BasicCostModel,
            'growing_cycle': 10,
        }
        greenhouse = GreenhouseNumpy(env_config)
        all_obs, all_actions = play_actions_on_numpy_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

        obs_shape = (len(Observation.FLAT_LABELS),)
        action_shape = (len(Action.LABELS),)

        for obs in all_obs:
            self.assertTrue(obs.shape == obs_shape, f"Obs shape must be {obs_shape}, not {obs.shape}")
        for action in all_actions:
            self.assertTrue(action.shape == action_shape, f"Obs shape must be {action_shape}, not {action.shape}")
