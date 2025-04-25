import math

import matplotlib.pyplot as plt
import numpy as np

from rl_greenhouse.greenhouse.physics import constants
from rl_greenhouse.greenhouse.types import State, Weather, Plant, Info
from rl_greenhouse.greenhouse.components.base import PlantComponent


class Lettuce(PlantComponent):
    """
    A lettuce plant simulation
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters,
            parameters_used={
                "shoot_mass": "The starting weight of the shoot in grams when the growing cycle starts.",
                "density_steps": "Planting density steps",
                "carbon_ppmv_max": "",
            },
        )
        self.shoot_mass = hyper_parameters["shoot_mass"]
        self.min_hours_needed_for_max_growth = hyper_parameters["min_hours_needed_for_max_growth"]
        self.growth_temp_optimal = hyper_parameters["growth_temp_optimal"]
        self.growth_temp_range = hyper_parameters["growth_temp_range"]
        self.growth_carbon_scalar = hyper_parameters["growth_carbon_scalar"]
        self.growth_par_scalar = hyper_parameters["growth_par_scalar"]
        self.max_daily_growth = hyper_parameters["max_daily_growth"]
        self.greenhouse_height = hyper_parameters["greenhouse_height"]

        self.max_price_plant = hyper_parameters["max_price_plant"]

        self.temperature_quality_loss_rate = hyper_parameters["temperature_quality_loss_rate"]
        self.temperature_tolerance_upper = hyper_parameters["temperature_tolerance_upper"]
        self.temperature_tolerance_lower = hyper_parameters["temperature_tolerance_lower"]

        self.humidity_tolerance_upper = hyper_parameters["humidity_tolerance_upper"]
        self.humidity_tolerance_lower = hyper_parameters["humidity_tolerance_lower"]
        self.humidity_quality_loss_rate = hyper_parameters["humidity_quality_loss_rate"]

        tipburn_tolerance_growing_rate_upper = hyper_parameters["tipburn_tolerance_growing_rate_upper"]
        self.tipburn_quality_loss_rate = hyper_parameters["tipburn_quality_loss_rate"]

        self.absorption_to_respiration_rate = hyper_parameters["absorption_to_respiration_rate"]
        self.mass_gain_to_co2_absorption_rate = hyper_parameters["mass_gain_to_co2_absorption_rate"]  # [kg/kg]
        self.respiration_rate = hyper_parameters["respiration_rate"]  # (kg/m²s)

        self.fixed_density_override_for_cost_model = hyper_parameters["fixed_density_override_for_cost_model"]
        self.density_steps = hyper_parameters["density_steps"]
        self.sell_overgrown_lettuce_for_max_price = hyper_parameters["sell_overgrown_lettuce_for_max_price"]
        self.carbon_ppmv_max = hyper_parameters["carbon_ppmv_max"]

        self.deficiency_severity = 0
        self.carbon_kg = 0
        self.growth_today = 1
        self.today_day_number = 0
        self.max_growth_a_second = self.max_daily_growth ** (1 / (3600 * self.min_hours_needed_for_max_growth))

        # As exponential growth factor per second
        self.tipburn_max_growth_rate = 1 + (self.max_growth_a_second - 1) * tipburn_tolerance_growing_rate_upper

    def current_selling_price(self, plant: Plant) -> float:
        """Returns the current selling price in € / plant"""
        if plant.mass >= 250:
            return self.max_price_plant * plant.quality
        return 0

        # dist_from_ideal_weight = abs(plant.mass - 250)
        #
        # if dist_from_ideal_weight > 40:
        #     return 0
        # elif dist_from_ideal_weight < 20:
        #     return (1 - 0.2 * dist_from_ideal_weight / 20) * plant.quality * self.max_price_plant
        # return (0.8 - 0.8 * ((dist_from_ideal_weight - 20) / 20)) * plant.quality * self.max_price_plant

    def is_first_step_of_the_day(self, time_current) -> bool:
        """

        :param time_current:
        :return:
        """
        now_day_number = math.floor(time_current / 24)
        if now_day_number > self.today_day_number:
            self.today_day_number += 1
            return True
        return False

    def step(self, info: Info, plant: Plant, state: State, weather: Weather, delta_time: float) -> (Plant, State, dict):
        """
        Calculates the resulting plant state from the current state in the greenhouse.

        :param info: The current extra information state
        :param plant: The current state of the plant
        :param state: The current state of the system
        :param weather: The weather at this time
        :param delta_time: Delta time in seconds
        """
        # Trigger daily growth limit reset
        if self.is_first_step_of_the_day(info.time):
            # print(self.growth_today)
            # print(self.growth_today)
            self.growth_today = 1

        # Maximum growth has been reached. Plant needs rest.
        if self.growth_today > self.max_daily_growth:
            return (
                plant,
                state,
                {
                    "damage_tipburn": float(0),
                    "damage_cold": float(0),
                    "damage_heat": float(0),
                    "damage_drought": float(0),
                    "damage_wet": float(0),
                },
            )

        # Apply resulting plant / state changes
        state, plant, growth_rate = self.grow(state, plant, delta_time)
        state, plant = self.respire(state, plant, delta_time)
        plant.plants_per_m2 = self.maximum_planting_density_for_plant_weight(plant.mass)
        plant, plant_damage_dict = self.quality(state, plant, growth_rate, delta_time)
        state.carbon_ppm = min(state.carbon_ppm, self.carbon_ppmv_max)
        return plant, state, plant_damage_dict

    def quality(self, state: State, plant: Plant, growth_rate: float, delta_time: float) -> (Plant, dict):
        """

        :param state:
        :param plant:
        :param growth_rate:
        :param delta_time:
        :return:
        """
        delta_hours = delta_time / 3600

        # Hourly quality loss rate in [1/h]
        quality_loss_rate = 0
        quality_loss_dict = {
            "damage_tipburn": float(0),
            "damage_cold": float(0),
            "damage_heat": float(0),
            "damage_drought": float(0),
            "damage_wet": float(0),
        }

        # Extreme cold / heat
        if state.temperature_air > self.temperature_tolerance_upper:
            degrees_over_limit = abs(self.temperature_tolerance_upper - state.temperature_air)
            quality_loss_dict["damage_heat"] = self.temperature_quality_loss_rate * degrees_over_limit
            quality_loss_rate += quality_loss_dict["damage_heat"]
        elif state.temperature_air < self.temperature_tolerance_lower:
            degrees_over_limit = abs(self.temperature_tolerance_lower - state.temperature_air)
            quality_loss_dict["damage_cold"] = self.temperature_quality_loss_rate * degrees_over_limit
            quality_loss_rate += quality_loss_dict["damage_cold"]

        # Drying out / Too much humidity
        if state.relative_humidity > self.humidity_tolerance_upper:
            percents_over_limit = (state.relative_humidity - self.humidity_tolerance_upper) * 100
            quality_loss_dict["damage_wet"] = self.humidity_quality_loss_rate * percents_over_limit
            quality_loss_rate += quality_loss_dict["damage_wet"]

        elif state.relative_humidity < self.humidity_tolerance_lower:
            percents_under_limit = (self.humidity_tolerance_lower - state.relative_humidity) * 100
            quality_loss_dict["damage_drought"] = self.humidity_quality_loss_rate * percents_under_limit
            quality_loss_rate += quality_loss_dict["damage_drought"]

        # Tipburn
        # Rapid growth rate causes lettuce tipburn due to calcium deficiency
        if growth_rate > self.tipburn_max_growth_rate or self.deficiency_severity > 0:
            # Can be negative if growth_rate < self.tipburn_tolerance_growing_rate_upper
            # Expressed in 1/s
            relative_growth_rate = (growth_rate - self.tipburn_max_growth_rate) / (self.tipburn_max_growth_rate - 1)

            deficiency_added = relative_growth_rate * 100 * delta_hours
            self.deficiency_severity = max(self.deficiency_severity + deficiency_added, 0)
            quality_loss_dict["damage_tipburn"] = self.tipburn_quality_loss_rate * self.deficiency_severity
            quality_loss_rate += quality_loss_dict["damage_tipburn"]

        quality_loss_rate = min(quality_loss_rate, 1)
        prev_quality = plant.quality
        plant.quality = plant.quality * (1 - quality_loss_rate) ** delta_hours

        quality_change = plant.quality - prev_quality
        total_damage_rate = sum(quality_loss_dict.values())
        if total_damage_rate != 0:  # If no damage is dealt, all values are 0.
            quality_loss_dict = {
                key: quality_change * val / total_damage_rate for key, val in quality_loss_dict.items()
            }
        return plant, quality_loss_dict

    def grow(self, state: State, plant: Plant, delta_time: float) -> (State, Plant):
        """

        :param state:
        :param plant:
        :param delta_time:
        :return:
        """
        growth, growth_rate = self._calculate_growth_rate(state, delta_time)
        mass_increase = plant.mass * (growth - 1)

        absorption_rate = plant.plants_per_m2 * self.mass_gain_to_co2_absorption_rate * mass_increase / delta_time

        state.carbon_ppm = self.modify_carbon_ppm(
            state.carbon_ppm, -absorption_rate, delta_time, self.greenhouse_height
        )

        # A part of the carbon can later be respired
        self.carbon_kg += absorption_rate * delta_time * self.absorption_to_respiration_rate

        plant.mass *= growth
        self.growth_today *= growth
        return state, plant, growth_rate

    def respire(self, state: State, plant: Plant, delta_time: float) -> (State, Plant):
        """

        :param state:
        :param plant:
        :param delta_time:
        :return:
        """
        plant_mass_per_square_meter = plant.mass / 1000 * plant.plants_per_m2  # In kg/m²
        resp_rate = self.respiration_rate * plant_mass_per_square_meter

        # Modify respiration if carbon in the plant is limited
        respired_carbon_kg = resp_rate * delta_time
        if respired_carbon_kg > self.carbon_kg:
            resp_rate = self.carbon_kg / delta_time
            self.carbon_kg = 0
        else:
            self.carbon_kg -= respired_carbon_kg

        # print(plant_mass_per_square_meter)
        state.carbon_ppm = self.modify_carbon_ppm(state.carbon_ppm, resp_rate, delta_time, self.greenhouse_height)
        return state, plant

    @staticmethod
    def modify_carbon_ppm(
        current_carbon_ppm: float, mass_flow_rate_in: float, delta_time: float, greenhouse_height: float
    ) -> float:
        """

        :param current_carbon_ppm:
        :param mass_flow_rate_in:
        :param delta_time:
        :param greenhouse_height:
        :return:
        """
        volume_flow_rate = mass_flow_rate_in / constants.DENSITY_CO2
        injected_co2_volume = delta_time * volume_flow_rate

        # Greenhouse volume per square meter = 1 x 1 x greenhouse height. PPMV increase below:
        ppmv_increase = 10e6 * injected_co2_volume / (injected_co2_volume + greenhouse_height)
        current_carbon_ppm += ppmv_increase
        return current_carbon_ppm

    def maximum_planting_density_for_plant_weight(self, weight: float) -> float:
        """
        Returns the maximum planting density to prevent quality loss in the lettuce.

        :param weight: The head fresh weight of the lettuce.
        """
        max_density_without_loss = 199.72499 * weight**-0.47616
        for density in self.density_steps:
            if max_density_without_loss > density:
                return density
        # print(max_density_without_loss)
        return 1

    def _calculate_growth_rate(self, state: State, delta_time: float) -> (float, float):
        """
        Returns the growth for this step and the growth rate in seconds
        """
        factor_temp = self._growth_factor_temperature(
            state.temperature_air, self.growth_temp_optimal, self.growth_temp_range
        )
        factor_light = self._growth_factor_light(state.par_light, self.growth_par_scalar)
        factor_co2 = self._growth_factor_co2(state.carbon_ppm, self.growth_carbon_scalar)

        assert 0 <= factor_temp <= 1, f"factor_temp must be 0 < f < 1, not {factor_temp}"
        assert 0 <= factor_light <= 1, f"factor_light must be 0 < f < 1, not {factor_light}"
        assert 0 <= factor_co2 <= 1, f"factor_co2 must be 0 < f < 1, not {factor_co2}"

        max_relative_growth = (self.max_growth_a_second**delta_time) - 1
        growth = 1 + max_relative_growth * factor_temp * factor_light * factor_co2

        # Apply growth
        new_growth_today = self.growth_today * growth
        if new_growth_today > self.max_daily_growth:
            growth = self.max_daily_growth / self.growth_today
            # print(f"Capping todays growth from {new_growth_today} to {self.max_daily_growth}")

        # Exponential Growth Rate in 1/s
        growth_rate = growth ** (1 / delta_time)
        return growth, growth_rate

    @staticmethod
    def _growth_factor_co2(carbon_ppm: float, growth_carbon_scalar: float) -> float:
        return 1.0 - np.exp(-carbon_ppm / growth_carbon_scalar)

    @staticmethod
    def _growth_factor_light(par_light: float, growth_par_scalar: float) -> float:
        return 1.0 - np.exp(-par_light / growth_par_scalar)

    @staticmethod
    def _growth_factor_temperature(
        temperature_air: float, growth_temp_optimal: float, growth_temp_range: float
    ) -> float:
        normalized_error = abs(temperature_air - growth_temp_optimal) / growth_temp_range
        return max(1 - (normalized_error) ** 2, 0)

    def reset(self) -> Plant:
        """
        Reset this subsystem.
        """
        self.carbon_kg = 0
        self.growth_today = 1
        self.today_day_number = 0
        return Plant(
            mass=self.shoot_mass,
        )


if __name__ == "__main__":
    from rl_greenhouse.greenhouse import DEFAULT_GREENHOUSE_CONFIG

    plt.figure()
    x = np.linspace(-20, 70, num=90)
    y = [Lettuce._growth_factor_temperature(xx, 25, 20) for xx in x]
    plt.grid()
    plt.xlabel("Air Temperature [°C]")
    plt.ylabel("Relative Growth Rate [-]")
    plt.plot(x, y)
    # plt.title("Temperature Growth")
    plt.show()

    plt.figure()
    x = np.linspace(300, 1200, num=90)
    y = [Lettuce._growth_factor_co2(xx, 400) for xx in x]
    plt.grid()
    plt.xlabel("Carbon PPMv [-]")
    plt.ylabel("Relative Growth Rate [-]")
    plt.plot(x, y)
    # plt.title("Carbon Growth")
    plt.show()

    plt.figure()
    x = np.linspace(0, 900, num=90)
    y = [Lettuce._growth_factor_light(xx, 250) for xx in x]
    plt.grid()
    plt.xlabel("Par Light [µmol/m²s]")
    plt.ylabel("Relative Growth Rate [-]")
    plt.plot(x, y)
    # plt.title("Light Growth")
    plt.show()

    plt.figure()
    x = np.linspace(0, 400, num=400)
    plt.grid()
    plt.xlabel("Fresh Weight [g]")
    plt.ylabel("Selling Price [€]")
    let = Lettuce(DEFAULT_GREENHOUSE_CONFIG)
    y = [let.current_selling_price(Plant(quality=1, mass=xx)) for xx in x]
    plt.plot(x, y)
    plt.show()
