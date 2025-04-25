import copy
import datetime
import math
import random
from typing import List, Dict, Tuple, Union, Type

import gym
import numpy as np
from gym.spaces import Box

from rl_greenhouse.greenhouse import reward_functions
from rl_greenhouse.greenhouse.config import DEFAULT_GREENHOUSE_CONFIG
from rl_greenhouse.greenhouse.types import Action, State, Observation, Info, Sequence, Plant, Weather, AdversarialAction
from rl_greenhouse.greenhouse.components.adversarial import AdversarialInputComponent
from rl_greenhouse.greenhouse.components.base import GreenhouseComponent, WeatherComponent, PlantComponent, EconomicComponent
from rl_greenhouse.greenhouse.reward_functions import RewardFunction

DEFAULT_STATE = State()


class Greenhouse(gym.Env):
    """
    Gym wrapper greenhouse simulator.

    Compatible with RLLib 1.8.0.

    It executes subsystems in order when they need to be fired.
    """

    # Badly defined for the default greenhouse environment as the input is not a numpy array.
    action_space = Box(
        low=np.array(
            [np.float32(bound[0]) for bound in list(Action.INTERVALS.values())],
        ),
        high=np.array(
            [np.float32(bound[1]) for bound in list(Action.INTERVALS.values())],
        ),
        dtype=np.float32,
    )

    observation_space = Box(
        low=np.array(
            [np.float32(bound[0]) for bound in list(Observation.INTERVALS.values())],
        ),
        high=np.array(
            [np.float32(bound[1]) for bound in list(Observation.INTERVALS.values())],
        ),
        dtype=np.float32,
    )

    name = "greenhouse"

    SUPPORTED_DOMAIN_RANDOMIZATION_PARAMS = []

    def __init__(self, config: dict):
        for key in config.keys():
            if key not in DEFAULT_GREENHOUSE_CONFIG.keys():
                raise KeyError(
                    f"Setting {key} is not contained in the default greenhouse configuration. "
                    f"Add to the default configuration to continue."
                    f"This error message prevents users from thinking they are adjusting a setting "
                    f"while in reality it is unrecognized and ignored."
                )

        # Set default values if they have not been overridden in the config dict!
        full_config = copy.deepcopy(DEFAULT_GREENHOUSE_CONFIG)
        full_config.update(config)

        self.full_config = full_config

        # Optional Domain Randomization
        self.domain_randomization_ranges = full_config["domain_randomization_ranges"]
        if self.domain_randomization_ranges is not None:
            if not isinstance(self.domain_randomization_ranges, dict):
                raise ValueError("Config parameter 'domain_randomization_ranges' should be None or dict.")
            for key in self.domain_randomization_ranges.keys():
                if key not in DEFAULT_GREENHOUSE_CONFIG.keys():
                    raise KeyError(f"Domain randomization config {key} does not exist in the Greenhouse config")

        # In case of domain randomization this is the currently used config
        self.current_domain_randomized_config: dict = ...

        # Adversarial Strength
        self.adversarial_strength = full_config["adversarial_strength"]

        # Params
        self.seed = 0
        self.simulation_delta_time_hours = full_config["sim_step_size_in_hours"]
        self.simulation_delta_time_seconds = self.simulation_delta_time_hours * 3600
        self.sim_steps_per_hour = int(1 / full_config["sim_step_size_in_hours"])
        self.growing_cycle = full_config["growing_cycle"]
        self.no_finish_on_mass_reached = full_config["no_finish_on_mass_reached"]

        reward_function: Union[str, Type[RewardFunction]] = full_config["reward_function"]
        if isinstance(reward_function, str):
            reward_function = getattr(reward_functions, reward_function)
        self.reward_function: RewardFunction = reward_function(full_config)
        self.reward_scale: float = full_config["reward_scale"]

        self.starting_info = full_config["starting_info"]
        self.starting_state = full_config["starting_state"]
        self.starting_weather = full_config["starting_weather"]
        self.starting_plant = full_config["starting_plant"]

        self.start_date = full_config["start_date"]
        self.start_date_randomization = full_config["start_date_randomization"]
        self.start_date_decade = full_config["start_date_decade"]
        self.start_date_offset = full_config["start_date_offset"]
        self.last_benchmark_start_year = 2010 + self.start_date_decade * 10

        # All system components
        self.greenhouse_model_components: [GreenhouseComponent] = [
            comp(full_config) for comp in full_config["greenhouse_model_components"]
        ]
        self.plant_model: PlantComponent = full_config["plant_model"](full_config)
        self.economic_model: EconomicComponent = full_config["economic_model"](full_config)
        self.weather_model: WeatherComponent = full_config["weather_model"](full_config)

        if full_config["adversarial_component"] is not None:
            self.adversarial_component: AdversarialInputComponent = full_config["adversarial_component"](full_config)

        # All state variables
        self.full_state: Observation = ...

        # Reset variables initially
        self.last_first_obs = ...
        self.reset()

    def step(self, action: Action, adversarial_action: AdversarialAction = None) -> (Observation, float, bool, dict):
        """
        Calculates the resulting observation from applying the effect of all subsystems.
        Tuple[Observation, float, Union[bool, Any], Dict[str, Union[float, Observation, Action]]]
        """
        full_state = self.full_state

        # Update weather time and economic model
        # Weather for time + 1 otherwise par light is calculated with old weather information which makes the
        # par light value incorrect. This shift fixes this by artificially first taking the weather w(t)
        # and then going into the update loop to predict s(t) as s(t) is dependent on w(t).
        full_state.weather = self.weather_model.step(full_state.info)

        # List of dicts with plant damage causes and their amounts
        plant_damage_dicts = []

        # Update plant model and greenhouse state
        for step in range(self.sim_steps_per_hour):
            is_final_step = step == self.sim_steps_per_hour - 1
            full_state.info.time += self.simulation_delta_time_hours
            if is_final_step:
                # This is done to prevent the time from being 23.99998 and the resulting hour being 23 instead of 0.
                full_state.info.time = round(full_state.info.time)
            full_state.info.hour = full_state.info.time % 24
            full_state.info.date_time += datetime.timedelta(hours=self.simulation_delta_time_hours)

            start_of_year = datetime.datetime(day=1, month=1, year=full_state.info.date_time.year)
            full_state.info.week = (full_state.info.date_time - start_of_year).days / 7

            full_state.plant, full_state.state, plant_damage_dict = self.plant_model.step(
                full_state.info,
                full_state.plant,
                full_state.state,
                full_state.weather,
                self.simulation_delta_time_seconds,
            )
            plant_damage_dicts.append(plant_damage_dict)

            # Update greenhouse components
            for component in self.greenhouse_model_components:
                full_state.state = component.step(
                    full_state.info, full_state.state, full_state.weather, action, self.simulation_delta_time_seconds
                )

            # In case of an adversarial action - Execute the adversarial_component
            if adversarial_action:
                full_state.state = self.adversarial_component.step(
                    full_state.info,
                    full_state.state,
                    full_state.weather,
                    adversarial_action.scale(self.adversarial_strength),  # Scaled with strength from config.
                    self.simulation_delta_time_seconds,
                )

            full_state.info = self.economic_model.step(
                full_state.info,
                full_state.plant,
                full_state.state,
                full_state.weather,
                action,
                self.plant_model.current_selling_price(full_state.plant),
                self.simulation_delta_time_seconds,
            )

        # print(full_state.plant.mass)
        if self.no_finish_on_mass_reached:
            done = full_state.info.time / 24 >= self.growing_cycle
        elif self.growing_cycle:
            done = full_state.info.time / 24 >= self.growing_cycle or full_state.plant.mass >= 250
        else:
            done = full_state.plant.mass >= 250
        reward = self.reward_function.calculate(full_state, done) * self.reward_scale
        obs = copy.deepcopy(self.full_state)

        # plant_damage_dict = dict(functools.reduce(operator.add, map(collections.Counter, plant_damage_dicts)))
        plant_damage_dict = {
            key: sum([dict_[key] for dict_ in plant_damage_dicts]) for key in plant_damage_dicts[0].keys()
        }

        info = {
            "obs": obs,
            "action": action,
            "costs_var_electricity": obs.info.costs_var_electricity,
            "costs_var_heat": obs.info.costs_var_heat,
            "costs_var_carbon": obs.info.costs_var_carbon,
            "costs_fix_total": obs.info.costs_fix_total,
            "costs_var_total": obs.info.costs_var_total,
            "average_head_per_m2": obs.info.average_head_per_m2,
            "final_balance": obs.info.balance,
            "quality": obs.plant.quality,
            **plant_damage_dict,
        }
        if math.isnan(reward):
            print("Reward NAN")
        self.full_state = full_state
        return obs, reward, done, info

    def _reset_all_simulation_components(self, start_day: datetime.datetime) -> (Info, Plant, State, Weather):
        # Reset all subsystems
        [sys.reset() for sys in self.greenhouse_model_components]

        self.economic_model.reset()

        # Resets the plant and weather model
        if self.starting_info:
            info = self.starting_info
        else:
            info = Info()
            info.date_time = start_day
            start_of_year = datetime.datetime(day=1, month=1, year=info.date_time.year)
            info.week = (start_day - start_of_year).days / 7

        if self.starting_weather:
            weather = self.starting_weather
        else:
            weather = self.weather_model.reset(info.date_time)

        if self.starting_plant:
            plant = self.starting_plant
        else:
            plant = self.plant_model.reset()

        if self.starting_state:
            state = self.starting_state
        else:
            state = State()

        self.reward_function.reset()
        return info, plant, state, weather

    def _sample_starting_day(self) -> datetime.datetime:
        """
        Randomly samples a starting day.
        """
        sample_first_year = 2001 + self.start_date_decade * 10
        sample_last_year = 2010 + self.start_date_decade * 10
        random_year = random.randint(sample_first_year, sample_last_year)

        if self.start_date_randomization is None:
            start_date = self.start_date

        elif self.start_date_randomization == "offset":
            random_offset = random.randint(-self.start_date_offset, self.start_date_offset)
            start_date = self.start_date + datetime.timedelta(days=random_offset)

        elif self.start_date_randomization == "year":
            start_date = datetime.datetime(random_year, self.start_date.month, self.start_date.day)

        elif self.start_date_randomization == "year_offset":
            random_offset = random.randint(-self.start_date_offset, self.start_date_offset)
            start_date = datetime.datetime(
                random_year, self.start_date.month, self.start_date.day
            ) + datetime.timedelta(days=random_offset)
        elif self.start_date_randomization == "full":
            max_days = (
                    datetime.datetime(year=sample_last_year, month=1, day=1)
                    - datetime.datetime(year=sample_first_year, month=1, day=1)
            ).days
            random_offset = random.randint(0, max_days)
            start_date = datetime.datetime(year=sample_first_year, month=1, day=1) + datetime.timedelta(
                days=random_offset
            )

        elif self.start_date_randomization == "benchmark":

            if self.last_benchmark_start_year == sample_last_year:
                benchmark_year = sample_first_year
            else:
                benchmark_year = self.last_benchmark_start_year + 1
            self.last_benchmark_start_year = benchmark_year
            # logging.debug(f"Starting benchmark for year {benchmark_year}")
            # print(f"Starting benchmark for year {benchmark_year}")
            start_date = datetime.datetime(benchmark_year, self.start_date.month, self.start_date.day)

        else:
            raise NotImplementedError(
                f"Start Date randomization method {self.start_date_randomization} is not implemented!"
            )
        return start_date

    def _randomize_config(self) -> dict:
        """Domain randomization config"""
        randomized_config = copy.deepcopy(self.full_config)
        for var, (low, high) in self.domain_randomization_ranges.items():
            randomized_config[var] = random.random() * (high - low) + low
        return randomized_config

    def _randomize_domain(self) -> None:
        """
        Resets the greenhouse components and sets their properties using the domain randomization.

        Reinitializes the following things:
            Reward function
            Plant model
            Greenhouse model components
            Economic model

        Note that the weather model is not adjusted. This is because it will have to load all weather types again.
        """
        config = self._randomize_config()
        self.current_domain_randomized_config = config

        reward_function: Union[str, Type[RewardFunction]] = config["reward_function"]
        if isinstance(reward_function, str):
            reward_function = getattr(reward_functions, reward_function)
        self.reward_function = reward_function(config)

        self.greenhouse_model_components: [GreenhouseComponent] = [
            comp(config) for comp in config["greenhouse_model_components"]
        ]
        self.plant_model: PlantComponent = config["plant_model"](config)
        self.economic_model: EconomicComponent = config["economic_model"](config)

    def reset(self) -> Observation:
        """
        Reset this subsystem.
        """
        if self.domain_randomization_ranges is not None and self.start_date_randomization != "benchmark":
            # print("Domain Randomization!")
            self._randomize_domain()

        start_date = self._sample_starting_day()
        info, plant, state, weather = self._reset_all_simulation_components(start_date)

        # Reset all values
        self.full_state = Observation(
            info=info,
            plant=plant,
            state=state,
            weather=weather,
        )
        self.last_first_obs = copy.deepcopy(self.full_state)
        return copy.deepcopy(self.full_state)

    def render(self, mode="human"):
        pass

    def __str__(self):
        return self.name


NORM_ACTION_SPACE_LOW = np.array([np.float32(0) for _ in list(Action.LABELS)])
NORM_ACTION_SPACE_HIGH = np.array([np.float32(1) for _ in list(Action.LABELS)])
NORM_OBS_SPACE_LOW = np.array([np.float32(0) for _ in list(Observation.FLAT_LABELS)])
NORM_OBS_SPACE_HIGH = np.array([np.float32(1) for _ in list(Observation.FLAT_LABELS)])


class GreenhouseNumpy(Greenhouse):
    """
    Numpy wrapper greenhouse simulator. In and outputs are bounded between 0 and 1.

    Compatible with RLLib 1.8.0.

    It executes subsystems in order when they need to be fired.
    """

    name = "greenhouse-numpy"

    action_space = Box(low=0, high=1, shape=(len(Action.LABELS),), dtype=np.float32)
    observation_space = Box(low=0, high=1, shape=(len(Observation.FLAT_LABELS),), dtype=np.float32)

    def __init__(self, config: dict):
        full_config = copy.deepcopy(DEFAULT_GREENHOUSE_CONFIG)
        full_config.update(config)
        self.clip_actions = full_config["clip_actions"]
        self.clip_output_within_bounds = full_config["clip_output_within_bounds"]
        super().__init__(full_config)

    def step(self, action: np.array, adversarial_action: np.array = None) -> (np.array, float, bool, dict):
        if self.clip_actions:
            action = np.clip(action, 0, 1)
            if adversarial_action is not None:
                adversarial_action = np.clip(adversarial_action, 0, 1)
        self._check_action(action)
        converted_action = Action.from_numpy(numpy_array=action).denormalize()
        if adversarial_action is not None:
            adversarial_action = AdversarialAction.from_numpy(adversarial_action).denormalize()

        obs, reward, done, info = super().step(converted_action, adversarial_action=adversarial_action)
        normalized_obs = obs.normalize().to_numpy()
        if self.clip_output_within_bounds:
            normalized_obs = normalized_obs.clip(0, 1)

        return normalized_obs, reward, done, info

    def reset(self) -> Observation:
        normalized_obs = super().reset().normalize().to_numpy()
        return normalized_obs

    def _check_action(self, normalized_action: np.array):
        self._check_object(normalized_action, Action.INTERVALS, "action")

    @classmethod
    def _check_object(cls, array: np.array, intervals: Dict[str, Tuple[float, float]], name: str):
        labels = list(intervals.keys())
        required_shape = (len(labels),)
        assert array.shape == required_shape, f"Shape should be {required_shape} not {array.shape}"

        array_mask = array < 0
        cls._check_extreme(array, array_mask, "at least 0", name, labels, intervals)

        array_mask = array > 1
        cls._check_extreme(array, array_mask, "at most 1", name, labels, intervals)

    @staticmethod
    def _check_extreme(
            array: np.array,
            array_mask: np.array,
            statement: str,
            name: str,
            labels: List[str],
            intervals: Dict[str, Tuple[float, float]],
    ):
        if np.any(array_mask):
            failing_labels = np.array(labels)[array_mask]
            failing_values = array[array_mask]
            failing_values_denormalized = {
                label: val * (intervals[label][1] - intervals[label][0]) + intervals[label][0]
                for label, val in zip(failing_labels, failing_values)
            }
            debug_info = "\n".join([f"{label}: {val}" for label, val in failing_values_denormalized.items()])
            raise ValueError(
                f"All {name} values must be {statement}, not to {np.min(array)}, "
                f"failing for labels {failing_labels}." + debug_info
            )

    def __str__(self):
        return self.name

    @classmethod
    def parse_obs_action_into_sequence(cls, all_obs: List[np.array], all_actions: List[np.array]) -> Sequence:
        regular_obs: List[Observation] = [Observation.from_n_values(obs).denormalize() for obs in all_obs]
        regular_actions: List[Action] = [Action(*action).denormalize() for action in all_actions]
        return Sequence(regular_obs, regular_actions)
