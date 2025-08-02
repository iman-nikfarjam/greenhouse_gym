"""
Rule based agent that actively controls the climate with feedback control.

This agent is based on heuristic growing information mainly provided by research at Cornell University.
A link to the research can be found here: https://cea.cals.cornell.edu/crops/.

It does not inherit from any RAY trainer as it is not meant to be run in parallel. It should be compatible with most
interfaces but there are no guarantees.
"""
from dataclasses import dataclass
from typing import Union, List

import numpy as np
try:  # pragma: no cover - optional dependency
    from ray.rllib import MultiAgentEnv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class MultiAgentEnv:  # type: ignore
        """Fallback stub when Ray is not installed."""
        pass

from rl_greenhouse.greenhouse.types import Action, Sequence, Observation
from rl_greenhouse.greenhouse import Greenhouse, GreenhouseNumpy, WrappedGreenhouse
from rl_greenhouse.greenhouse.env_wrappers import WrappedGreenhouseMultiAgent

ObsType = Union[Observation, np.ndarray]
ActionType = Union[Action, np.ndarray]


@dataclass
class ControllerSettings:
    """
    Settings for the RuleBased Agent.

    These settings have been optimized on the greenhouse benchmark standards of 2-Feb-2022, optimizing final balance
    on the first decade 2001-2010. This results in a final balance on the real benchmark decade of 2011-2020 of €/m².

    """
    carbon_level_closed: float = 676.07
    carbon_level_night: float = 555.08
    carbon_level_open: float = 423.5
    carbon_p_action: float = 0.01
    day_start_hour: float = 4.2695
    heater_minimal_temp_day: float = 24.0
    heater_minimal_temp_night: float = 23.03
    illumination_max_watts: float = 1048.7
    light_start: float = 7.0
    light_stop: float = 7.0
    minimal_difference_to_open: float = 0.086096
    night_start_hour: float = 21.92
    p_control_blackout: float = 0.017092
    p_control_heater: float = 0.2
    ventilation_humidity_p_action: float = 0.25
    ventilation_max_humidity: float = 0.8887
    ventilation_max_temp: float = 25.292
    ventilation_min_humidity: float = 0.6
    ventilation_p_action: float = 0.39137


CORNELL_SETPOINTS = ControllerSettings(
    heater_minimal_temp_day=24,
    heater_minimal_temp_night=19,
    ventilation_min_humidity=0.5,
    ventilation_max_humidity=0.7,
    minimal_difference_to_open=0,
    carbon_level_night=390,
    carbon_level_closed=1500,
    carbon_level_open=1500,
    day_start_hour=5,
    night_start_hour=20,
    light_start=6,
    light_stop=7,
)


class RuleBased:
    """
    Rule Based PD-controller for controlling the greenhouse.

    :param env: The environment from which the action / observation space are inferred.
    :param controller_settings: The settings for control algorithm within this class.
    """

    def __init__(self, env: Greenhouse, controller_settings=ControllerSettings()):
        self.env = env

        self.set = controller_settings

        self.multi_env = isinstance(env, MultiAgentEnv)

        if isinstance(env, (GreenhouseNumpy, WrappedGreenhouse, WrappedGreenhouseMultiAgent)):
            self.continuous_action = True
            self.normalized_action = True
            self.normalized_observation = True
            self.type_action = np.ndarray
            self.type_observation = np.ndarray

        elif isinstance(env, Greenhouse):
            self.continuous_action = True
            self.normalized_action = False
            self.normalized_observation = False
            self.type_action = Action
            self.type_observation = Observation

        else:
            raise NotImplementedError(f"Env of Type {env.__class__} is not supported!")

    def step(self) -> (Sequence, float):
        """
        Plays one episode in the Greenhouse Simulator
        """
        done = False
        obs = self.env.reset()
        all_obs = [obs]
        all_actions = []
        all_infos = []
        total_reward = 0
        final_balance = 0
        while not done:
            action = self.compute_action(obs)
            all_actions.append(action)
            obs, reward, done, info = self.env.step(action)
            all_obs.append(obs)
            all_infos.append(info)
            total_reward += reward
            final_balance = info["final_balance"]

        return self._parse_into_sequence(all_obs, all_actions, all_infos), total_reward, final_balance

    def get_policy(self):
        return self

    def compute_actions(self, obs_batch: np.ndarray, **kwargs) -> List[ActionType]:
        return [np.stack([self.compute_action(obs_batch[i, :]) for i in range(obs_batch.shape[0])])]

    def compute_action(self, obs: ObsType) -> ActionType:
        """
        Computes an action from the current observation.
        :param obs:
        """
        if self.multi_env:
            obs = obs['protagonist']

        obs = self._parse_obs_to_denormalized_observation_object(obs)

        if self.continuous_action:
            normalized_action = self._compute_continuous_action(obs)
        else:
            normalized_action = self._compute_discrete_action(obs)

        return self._parse_normalized_action_to_correct_action_type(normalized_action)

    def compute_single_action(self, observation: ObsType = None, **kwargs):
        return self.compute_action(observation)

    def _compute_continuous_action(self, obs: Observation) -> Action:
        normalized_action = Action()

        info, plant, state, weather = obs.unpack()

        day = self.set.night_start_hour > obs.info.hour > self.set.day_start_hour

        if day:
            if state.temperature_air < self.set.heater_minimal_temp_day:
                error = self.set.heater_minimal_temp_day - state.temperature_air
                normalized_action.heater_power = max(min(error * self.set.p_control_heater, 1), 0)
        else:
            if state.temperature_air < self.set.heater_minimal_temp_night:
                error = self.set.heater_minimal_temp_night - state.temperature_air
                normalized_action.heater_power = max(min(error * self.set.p_control_heater, 1), 0)

        if state.temperature_air > self.set.ventilation_max_temp:
            error = state.temperature_air - self.set.ventilation_max_temp
            normalized_action.ventilation_position = min(error * self.set.ventilation_p_action, 1)

        if day:
            if weather.illumination_out > self.set.illumination_max_watts:
                error = weather.illumination_out - self.set.illumination_max_watts
                normalized_action.screen_blackout_position = min(error * self.set.p_control_blackout, 1)
        else:
            normalized_action.screen_blackout_position = 1
            normalized_action.screen_transparent_position = 1

        if state.relative_humidity > self.set.ventilation_max_humidity and \
                (state.relative_humidity - weather.relative_humidity_out) > self.set.minimal_difference_to_open:
            error = state.relative_humidity - self.set.ventilation_max_humidity
            ventilation_humidity = min(error * self.set.ventilation_humidity_p_action, 1)
            normalized_action.ventilation_position = max(normalized_action.ventilation_position, ventilation_humidity)

        if state.relative_humidity < self.set.ventilation_min_humidity and \
                (weather.relative_humidity_out - state.relative_humidity) > self.set.minimal_difference_to_open:
            error = self.set.ventilation_min_humidity - state.relative_humidity
            ventilation_humidity = min(error * self.set.ventilation_humidity_p_action, 1)
            normalized_action.ventilation_position = max(normalized_action.ventilation_position, ventilation_humidity)

        if day:
            if normalized_action.ventilation_position == 0:
                carbon_setpoint = self.set.carbon_level_closed
            else:
                carbon_setpoint = self.set.carbon_level_open
        else:
            carbon_setpoint = self.set.carbon_level_night

        if state.carbon_ppm < carbon_setpoint:
            error = carbon_setpoint - state.carbon_ppm
            normalized_action.co2_flow_rate = max(min(error * self.set.carbon_p_action, 1), 0)

        if self.set.light_start < info.hour < self.set.light_stop:
            normalized_action.enable_lamps = 1

        return normalized_action

    def _parse_into_sequence(
            self, all_obs: List[ObsType], all_actions: List[ActionType], all_infos: List[dict]
    ) -> Sequence:
        if self.type_observation == np.ndarray:
            all_obs = [Observation.from_n_values(*nd_array) for nd_array in all_obs]
        if self.type_action == np.ndarray:
            all_actions = [Action.from_numpy(nd_array) for nd_array in all_actions]
        if self.normalized_observation:
            all_obs = [obs.denormalize() for obs in all_obs]
        return Sequence(all_obs, all_actions, env_infos=all_infos)

    def _compute_discrete_action(self, obs: Observation) -> Action:
        """

        :param obs:
        :return:
        """
        action_dict = self._compute_continuous_action(obs).to_dict()
        for i, (key, val) in enumerate(action_dict.items()):
            action_dict[key] *= (self.env.action_space[i].n - 1)
            action_dict[key] = round(action_dict[key])
        return Action(**action_dict)

    def _parse_obs_to_denormalized_observation_object(self, obs: ObsType) -> Observation:
        if self.type_observation == np.ndarray:
            obs = Observation.from_n_values(*obs)
        if self.normalized_observation:
            obs = obs.denormalize()
        return obs

    def _parse_normalized_action_to_correct_action_type(self, normalized_action: Action) -> ActionType:
        if self.continuous_action and not self.normalized_action:
            action = normalized_action.denormalize()
        else:
            action = normalized_action

        if self.type_action == np.ndarray:
            return action.to_numpy()
        return action

    def __str__(self):
        return "RuleBased"
