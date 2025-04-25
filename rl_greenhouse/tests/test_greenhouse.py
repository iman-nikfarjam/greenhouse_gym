import unittest

from rl_greenhouse.greenhouse.types import Action
from rl_greenhouse.greenhouse.env import Greenhouse
from rl_greenhouse.greenhouse.components.economic import BasicCostModel
from rl_greenhouse.greenhouse.components.greenhouse import Co2Supply, StandardLights, WallConduction, Ventilation, Heater
from rl_greenhouse.greenhouse.components.plant import NoPlant, Lettuce
from rl_greenhouse.greenhouse.components.weather import HoekVanHollandWeatherStation, ConstantWeatherStation
from rl_greenhouse.utils.evaluation import play_actions_on_env


class GreenhouseTests(unittest.TestCase):
    """
    To test:

    """

    CONSTANT_ACTION = constant_action = Action(
        heater_power=23,
        co2_flow_rate=1e-6,
        enable_lamps=0,
        ventilation_position=0,
        screen_transparent_position=0.6,
    )

    CONSTANT_ACTIONS_10_DAYS = [CONSTANT_ACTION] * 10 * 24

    def test_constant_action(self):
        env_config = {
            'greenhouse_model_components': [],
            'weather_model': ConstantWeatherStation,
            'plant_model': NoPlant,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_hoek_van_holland_weather(self):
        env_config = {
            'greenhouse_model_components': [],
            'weather_model': HoekVanHollandWeatherStation,
            'plant_model': NoPlant,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_plant(self):
        env_config = {
            'greenhouse_model_components': [],
            'weather_model': ConstantWeatherStation,
            'plant_model': Lettuce,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_co2_supply(self):
        env_config = {
            'greenhouse_model_components': [Co2Supply],
            'weather_model': ConstantWeatherStation,
            'plant_model': NoPlant,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_heater(self):
        env_config = {
            'greenhouse_model_components': [Heater],
            'weather_model': ConstantWeatherStation,
            'plant_model': NoPlant,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_lights_subsystem(self):
        env_config = {
            'greenhouse_model_components': [StandardLights],
            'weather_model': ConstantWeatherStation,
            'plant_model': NoPlant,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_lights_ventilation(self):
        env_config = {
            'greenhouse_model_components': [Ventilation],
            'weather_model': ConstantWeatherStation,
            'plant_model': NoPlant,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_lights_conduction(self):
        env_config = {
            'greenhouse_model_components': [WallConduction],
            'weather_model': ConstantWeatherStation,
            'plant_model': NoPlant,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_cost_model(self):
        env_config = {
            'greenhouse_model_components': [],
            'weather_model': ConstantWeatherStation,
            'plant_model': NoPlant,
            'economic_model': BasicCostModel,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)

    def test_full_system(self):
        env_config = {
            "greenhouse_model_components": [Heater, WallConduction, StandardLights, Ventilation, Co2Supply],
            "weather_model": HoekVanHollandWeatherStation,
            "plant_model": Lettuce,
            'economic_model': BasicCostModel,
            'growing_cycle': 10,
        }
        greenhouse = Greenhouse(env_config)
        play_actions_on_env(greenhouse, self.CONSTANT_ACTIONS_10_DAYS, stop_when_done=False)


class GreenhouseReplayTests(unittest.TestCase):
    """
    To test:

    """
    REPLAY_STATES = ...
    REPLAY_WEATHER = ...
    REPLAY_PLANT = ...

