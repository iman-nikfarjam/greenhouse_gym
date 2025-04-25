from rl_greenhouse.greenhouse.physics import constants
from rl_greenhouse.greenhouse.types import State, Weather, Action, Info, Plant
from rl_greenhouse.greenhouse.components.base import EconomicComponent


class BasicCostModel(EconomicComponent):
    """
    A simple cost model that includes variable electricity prices, a price for heat and carbon.
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters,
            parameters_used={
                "lamp_intensity": "",
                "price_electricity_on_peak": "Price in € / kWh for electricity during peak hours",
                "price_electricity_off_peak": "Price in € / kWh for electricity during off peak hours",
                "price_heating": "Price in € / MJ for heating",
                "price_carbon": "The price of a kilogram of carbon",
            },
        )
        self.price_var_electricity_on_peak = hyper_parameters["price_electricity_on_peak"]
        self.price_var_electricity_off_peak = hyper_parameters["price_electricity_off_peak"]
        self.price_var_heating = hyper_parameters["price_heating"]
        self.price_var_carbon = hyper_parameters["price_carbon"]

        self.fixed_costs = hyper_parameters["fixed_costs"]
        self.fixed_density_override_for_cost_model = hyper_parameters["fixed_density_override_for_cost_model"]

        self.sum_of_1_over_density = 0
        self.nr_densities = 0

    def reset(self):
        self.sum_of_1_over_density = 0
        self.nr_densities = 0

    def step(
        self,
        info: Info,
        plant: Plant,
        state: State,
        weather: Weather,
        action: Action,
        current_selling_price: float,
        delta_time: float,
    ) -> Info:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param plant: The current state of the plant
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param current_selling_price: The current selling price per plant
        :param delta_time: Delta time in seconds
        """
        if self.fixed_density_override_for_cost_model:
            info.average_head_per_m2 = self.fixed_density_override_for_cost_model
        else:
            info = self._update_average_head_m2(info, plant)
        info = self._set_prices(info)

        info = self._update_variable_costs(info, state, action, delta_time)
        info = self._update_fixed_costs(info, delta_time)
        info.gains_plant = current_selling_price * info.average_head_per_m2

        info.balance = info.gains_plant - info.costs_var_total - info.costs_fix_total
        return info

    def _update_average_head_m2(self, info: Info, plant: Plant) -> Info:
        self.sum_of_1_over_density += 1 / plant.plants_per_m2
        self.nr_densities += 1
        info.average_head_per_m2 = self.nr_densities / self.sum_of_1_over_density
        return info

    def _update_variable_costs(self, info: Info, state: State, action: Action, delta_time: float) -> Info:
        """Uses the current state to update the variable costs."""
        electricity_usage = (state.electric_power * delta_time) / constants.JOULES_PER_KWH
        heat_usage = (action.heater_power * delta_time) / 10e6
        carbon_usage = action.co2_flow_rate * delta_time

        info.cum_electricity_usage += electricity_usage
        info.cum_heat_usage += heat_usage
        info.cum_carbon_usage += carbon_usage

        info.costs_var_electricity += electricity_usage * info.price_electricity
        info.costs_var_heat += heat_usage * info.price_heat
        info.costs_var_carbon += carbon_usage * info.price_carbon
        info.costs_var_total = info.costs_var_electricity + info.costs_var_heat + info.costs_var_carbon
        return info

    def _update_fixed_costs(self, info: Info, delta_time: float) -> Info:
        """"""
        fraction_of_year = delta_time / constants.SECONDS_IN_A_YEAR
        info.costs_fix_total += self.fixed_costs * fraction_of_year
        return info

    def _set_prices(self, info: Info) -> Info:
        """

        :param info:
        """
        if 7 <= info.hour < 23:
            info.price_electricity = self.price_var_electricity_on_peak
        else:
            info.price_electricity = self.price_var_electricity_off_peak

        info.price_heat = self.price_var_heating
        info.price_carbon = self.price_var_carbon
        return info
