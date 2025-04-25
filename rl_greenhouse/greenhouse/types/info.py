import datetime
from dataclasses import dataclass

from rl_greenhouse.greenhouse.types.base import BaseDataClass


@dataclass
class Info(BaseDataClass):
    """
    The default meta information interface

    time: Hours into the simulation.
    """

    time: float = 0  # Hours since start of growing cycle
    week: float = 0  # Week nr since 1st of Jan
    hour: float = time % 24  # Current hour of the day

    cum_electricity_usage: float = 0  # In kWh/m²
    cum_heat_usage: float = 0  # In MJ/m²
    cum_carbon_usage: float = 0  # kg/m²

    price_electricity: float = 0.088  # €/(kWh/m²)
    price_heat: float = 0.01341  # €/(MJ/m²)
    price_carbon: float = 0.14  # €/(kg/m²)

    costs_var_electricity: float = 0  # €/m²
    costs_var_heat: float = 0  # €/m²
    costs_var_carbon: float = 0  # € /m²

    # €/m²
    average_head_per_m2: float = 25  # Plants per m²
    gains_plant: float = 0
    costs_var_total: float = costs_var_electricity + costs_var_heat + costs_var_carbon
    costs_fix_total: float = 0
    balance: float = gains_plant - costs_var_total - costs_fix_total

    # Not a float. Not included in the to_numpy function
    date_time: datetime.datetime = None

    INTERVALS = {
        "time": [0, 2400],
        "week": [0, 52],
        "hour": [0, 24],
        "cum_electricity_usage": [0, 2400],
        "cum_heat_usage": [0, 2400],
        "cum_carbon_usage": [0, 10],
        "price_electricity": [0, 1],
        "price_heat": [0, 10],
        "price_carbon": [0, 10],
        "costs_var_electricity": [0, 10],
        "costs_var_heat": [0, 10],
        "costs_var_carbon": [0, 10],
        "average_head_per_m2": [0, 100],
        "gains_plant": [0, 50],
        "costs_var_total": [0, 50],
        "costs_fix_total": [0, 50],
        "balance": [-50, 50],
    }
    LABELS = list(INTERVALS.keys())


if __name__ == "__main__":
    info_state = Info()
    np_info = info_state.to_numpy_2d()
    norm_info = info_state.normalize()
    denormalized_info = norm_info.denormalize()
    print(info_state)
    print(np_info)
    print(norm_info)
    print(denormalized_info)
