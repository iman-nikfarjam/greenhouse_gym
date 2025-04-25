import datetime
from typing import Dict, Any

from rl_greenhouse.greenhouse import reward_functions
from rl_greenhouse.greenhouse.components.adversarial import AdversarialInputComponent
from rl_greenhouse.greenhouse.components.economic import BasicCostModel
from rl_greenhouse.greenhouse.components.greenhouse import Heater, WallConduction, StandardLights, Ventilation, Co2Supply, \
    HydroponicsBasin
from rl_greenhouse.greenhouse.components.plant import Lettuce
from rl_greenhouse.greenhouse.components.weather import HoekVanHollandWeatherStation
from rl_greenhouse.greenhouse.types import State

DEFAULT_GREENHOUSE_CONFIG: Dict[str, Any] = {
    # ================================ Greenhouse Numpy Config Section ================================
    # True to clip actions in the -1, 1 range.
    "clip_actions": True,
    # True to clip output in the -1, 1 range.
    "clip_output_within_bounds": True,

    # ======================================== Wrapper Config =========================================

    # ====== Wrapped Greenhouse Config ======
    # What wrappers to use. Should be a list of wrapper types.
    # The leftmost wrapper is applied first and the rightmost wrapper is applied last.
    # The first item of the list should be of gym.Env type (Typically WrappedGreenhouse).
    # Must use the WrappedGreenhouse class, otherwise this setting is ignored.
    "wrappers": None,
    # === GaussianNoiseWrapper Only ===
    # Mu for gaussian noise applied. Adds Bias to the observation.
    "gaussian_noise_mu": 0,
    # Standard deviation for gaussian noise applied. Adds Variance to the observation.
    "gaussian_noise_std": 0.01,
    # === FrameStackWrapper Only ===
    # Frame stacking. Only works when any FrameStackWrapper is used!
    "nr_frames_stacked": 6,
    # === AdversarialWrappers Only ===
    # Adversarial Component. Only works when the AdversarialWrapper is enabled.
    "adversarial_component": AdversarialInputComponent,
    # Adversarial Action Strength. Defaults to 1. Adversarial action is multiplied with this factor.
    # E.g. adversarial heating of -5 with strength 1.5 -> -7.5
    "adversarial_strength": 1,
    # Deprecated
    "adversary_locked": None,  # Locked version of the adversarial agent. One of the two should be locked.
    "agent_locked": None,  # Locked version of the regular agent. One of the two should be locked.
    "type_locked": None,  # Defines the Trainer class type for both locked options
    # === Parametric Domain Randomization Only ===
    # Optional domain randomization that randomizes the greenhouse parameters upon reset of the environment.
    # Should be a dict mapping greenhouse config parameter to a tuple between which uniform sampling is performed for
    # domain randomization. Can modify any number of greenhouse parameters for the plant model and the greenhouse
    # components.
    # Example:
    #   'domain_randomization_ranges': {'reflectance_greenhouse': (0.4, 1.0)}
    # Note 1: The weather model cannot be varied.
    # Note 2: If benchmark mode is enabled, domain randomization will not be applied to keep the benchmark
    #   deterministic. This setting will then be ignored.
    "domain_randomization_ranges": None,
    # === ReplayState / ReplayPlant / ReplayWeather Components Only ===
    # In case of using ReplayState, set to a list of State objects to replay.
    "states_to_replay": None,
    # In case of using ReplayPlant, set to a list of Plant objects to replay.
    "plants_to_replay": None,
    # In case of using ReplayWeather, set to a list of Weather objects to replay.
    "weather_to_replay": None,
    # Variables to blacklist from the replay component. These variables will be ignored by these components.
    "variable_blacklist": [],
    # ================================ Greenhouse Simulation Settings ================================
    # All the components that are simulated in the greenhouse. Order determines order of execution on each time step.
    "greenhouse_model_components": [Heater, WallConduction, StandardLights, Ventilation, Co2Supply, HydroponicsBasin],
    # The plant model used in the simulation.
    "plant_model": Lettuce,
    # The weather model used in the simulation.
    "weather_model": HoekVanHollandWeatherStation,
    # The economic model used in the simulation.
    "economic_model": BasicCostModel,
    # The step size for a single step (in hours).
    "sim_step_size_in_hours": 5 / 60,
    # === Starting Conditions ===
    # Starting date of the simulator.
    "start_date": datetime.datetime(year=2001, month=3, day=1),
    # Randomizes the start date. Either None, 'year', 'full'
    # - None : No randomization
    # - 'offset'        : Uniformly sample from start_date - offset to start_date + offset
    # - 'year'          : Randomly select different starting years (But the same month / day)
    # - 'year_offset'   : Uniformly sample from start_date - offset to start_date + offset with a random starting year
    # - 'full'          : Completely random starting date
    # - 'benchmark'     : Repeating starting date cycle. Starts sequentially in all years from 2011 to 2020 at set d/m
    "start_date_randomization": "year_offset",
    # Sampling Decade 0 for 2001 to 2010, 1 for 2011 to 2020 when sampling years.
    # Only works for 'year', 'year_offset' and 'benchmark'
    "start_date_decade": 0,
    # Offset variable used in random start date selection. Start Date in days +- start date offset
    "start_date_offset": 30,
    # Info at the start of the simulator, None indicates default Info constructor is used.
    "starting_info": None,
    # The starting weather, None indicates reset of weather model is used.
    "starting_weather": None,
    # The starting state of the system, None indicates the default State constructor is used.
    "starting_state": None,
    # Starting state of the plant, None indicates reset of plant model is used.
    "starting_plant": None,
    # === Stopping Conditions ===
    # The max growing cycle duration in days.
    "growing_cycle": 240,
    # Set to true to not return done when the selling mass is reached.
    "no_finish_on_mass_reached": False,
    # ====== Gym Environment Settings ======
    # Gym environment specifics
    # The reward function to determine the step reward. Must be of RewardFunction class.
    "reward_function": reward_functions.DailyBalanceReward,
    # The scaling value for the reward function. Scales the reward by multiplying with this value.
    "reward_scale": float(1.0),
    # === Practical Limits ===
    # Maximum CO2 level in the system.
    "carbon_ppmv_max": State.INTERVALS["carbon_ppm"][1],
    # ================================ Greenhouse Simulation Parameters ================================
    # The height of the greenhouse in [m]
    "greenhouse_height": 3,
    # The heat capacity of the greenhouse in [J/m² K]
    # Double from Wouter Kuijpers thesis, this is done to allow the agent to have some control as we use
    # a step size of 1 hour. Otherwise, a small miscalculation in action would result in huge temperature changes.
    "greenhouse_heat_capacity": 60_000,  # 1_000_000
    # === Heater Model ===
    # Heat exchange between heating pipes and greenhouse in [W/m²K]
    "pipe_heat_exchange_coefficient": 1.5,
    # Efficiency of the heating system [-]
    "heating_efficiency": 0.8,
    # === Reflectance Model ===
    # What fraction of the sunlight is reflected by the greenhouse in [-]
    "reflectance_greenhouse": 0.20,
    # What fraction of the sunlight is reflected by the blackout screen if fully enabled in [-]
    "reflectance_blackout": 0.86,
    # What fraction of the sunlight is reflected by the transparent screen if fully enabled in [-]
    "reflectance_transparent": 0.27,
    # The amount of heat from sun that is trapped by the greenhouse system between 0 and 1
    "fraction_sun_heat_absorbed": 0.28,
    # The amount of heat from the sun that is blocked by the screens. The remaining part is absorbed regardless.
    "heat_blocked_by_screens": 0.7,
    # === Lights ===
    # The intensity of the lamp in par light [µmol/m² s]
    # 250 - Recommended value found in Cornel University
    # 100 - Average greenhouse analysis found in Onderzoek+Marktpotentieel+Glastuinbouw+VS.pdf (See below)
    "lamp_intensity": 200,
    # Convert intensity of lamp in par light to Watts consumed
    # https://edepot.wur.nl/156931
    "lamp_par_to_watts": 0.185,
    # === Economic Model ===
    # Whether to sell lettuce over 250 grams for the maximum price.
    # Useful to compare performance when the growing cycle is fixed.
    "sell_overgrown_lettuce_for_max_price": True,
    # Maintenance costs
    # Costs made per m² per year due to all
    # (2343200 - 582700) / 62000
    # Source: Onderzoek Marktpotentieel Glastuinbouw VS - Table 3.1
    # https://www.agroberichtenbuitenland.nl/binaries/agroberichtenbuitenland/documenten/rapporten/2020/07/29/
    # tuinbouwrapport-vs/2020-054+Onderzoek+Marktpotentieel+Glastuinbouw+VS.pdf
    "fixed_costs": 28.40,
    # https://www.cbs.nl/en-gb/news/2018/16/upscaling-of-greenhouse-vegetable-production
    # 2017 - 4990 ha greenhouses /4.0 / greenhouse -> 1248 greenhouses
    # https://www.kasalsenergiebron.nl/content/user_upload/Energiemonitor_Nederlandse_glastuinbouw_2017.pdf
    # Total electricity of greenhouses in 2017 -> 5.5 Mton
    # Using file:///home/woutervdb/Documents/papers_new/bijlage-bij-kamerbrief-toezending-wur-rapport-energiemonitor-van-de-nederlandse-glastuinbouw-2020.pdf
    # Figure 4.4 - 7.5 Miljard kwh -> 6.0096 million kwh / greenhouse
    # 150 kwh per m² per year
    # Also from graph 6.1 prod matches 102 kwh per m²
    # Then 7.5 - 125 m²
    # Per greenhnouse of 4 acres -> 5M kwh per year -> 5000 mwh
    # Price 0.141 from CBS
    # https://edepot.wur.nl/51443
    # Difference in dal - plateau 1.9188960856096875 - factor 1.9
    # Split evenly across on / off-peak results in values below:
    # Price in € / kWh for electricity during peak hours (Between 7:00 and 23:00)
    "price_electricity_on_peak": 0.168,  #
    # Price in € / kWh for electricity during off-peak hours
    "price_electricity_off_peak": 0.088,
    # Price in € / MJ for heating
    # https://opendata.cbs.nl/statline/#/CBS/nl/dataset/81309NED/table?fromstatweb  - 13,410
    # Category 10-100 TJ / y
    # https://edepot.wur.nl/534626 - 1100 - 1900 MJ / m²
    # https://www.cbs.nl/en-gb/news/2018/16/upscaling-of-greenhouse-vegetable-production - 4.0 ha per greenhouse
    # 40_000 * 1100 = 44 TJ
    "price_heating": 0.01341,  # € / MJ  (170 EUR / MWH, end 2022)
    # Carbon dioxide sequestration by mineral carbonation
    # https://edepot.wur.nl/121870
    # Sequestration from other industries 77 - 102 € / ton
    # The price of a kilogram of carbon
    # Is in range of KWIN - 0.08 - 0.14
    # https://edepot.wur.nl/296417 also confirms this.
    "price_carbon": 0.14,
    # Maximum selling price per plant
    # Dutch horticulture source for avg greenhouse profit
    # (2_800_700 / 62_000)/ 12 / 25 = 0.15 on average.
    # We expect grower to grow at 75% quality, therefore at 100% quality the price becomes 0.20.
    # Perfect plant grown in perfect conditions should be able to earn almost double this amount, therefore 25 cts.
    "max_price_plant": 0.20,
    # === Ventilation ===
    # Heat exchange in W/m²K with an open window
    # Vent capacity of 50 -> 16.66 exchanges of 1200 KJ / K / h -> 5.55 W/m²/k
    # Next to that we have radiation outgoing from the system.
    # This is assumed to be approx 5 watts
    # Based on this source: However: This is a bad approximation as it does not match what they say in the source.
    # https://asterism.org/wp-content/uploads/2019/03/tut37-Radiative-Cooling.pdf
    "heat_exchange_open_window": 10,
    # Max ventilation capacity with fully open windows in [m3/h]
    # About 9-26.5 x or 3 to 9.5 full air refreshes an hour for the other vents.
    # Source: Analysis of Greenhouse Ventilation Efficiency based on Computational Fluid Dynamics
    "max_ventilation_capacity": 50,
    # === Hydroponics Basin ===
    "water_evaporation_coefficient_base": 26,
    "water_evaporation_coefficient_wind": 19,
    "water_evaporation_heat": 2_453_500,  # J / kg @ 20 deg
    "water_exposed_area_per_m2": 0.04,
    "wall_temp_fraction_inside_air": 0.9,
    "volume_flow_by_wall": 0.002,
    # === Conduction ===
    # Has been fit to data
    # Old Sources
    # Single sheet of Polyethylene has a value of 0.85
    # https://www.globalplasticsheeting.com/our-blog-resource-library/topic/solawrap
    # Window R value of 0.92 https://www.coloradoenergy.org/procorner/stuff/r-values.htm
    # Plant factories vs greenhouses paper R - value of 0.88
    # H value in [W/k] for the greenhouse roof. Based on conductance of 0.92 for
    "roof_heat_transfer_rate": 1.08,
    # Increases enthalpy by this factor scaled by outside wind speed in [W/(km/s)]
    "roof_h_wind_effect": 0.20,
    # R value in [W/k] for the greenhouse blackout screen if enabled
    # Plant factories vs greenhouses paper R - value of 0.87
    "screen_blackout_insulation": 0.87,
    # R value in [W/k] for the greenhouse transparent screen if enabled
    # Plant factories vs greenhouses paper R - value of 0.87
    "screen_transparent_insulation": 0.87,
    # === Plant Growth ===
    # The starting weight of the shoot in grams when the growing cycle starts.
    "shoot_mass": 2.26,
    # Plant growth parameters have been fitted on plant growth data. These parameters minimize loss.
    # Optimal Growth Temp: 24 file:///home/woutervdb/Documents/papers_new/WPR-1049_LR.pdf
    # Complete guide for growing lettuce hydroponically
    "growth_temp_optimal": 25,
    # Revised on 26-Jan-2022 using demo_8_fit_entire_plant_revision_jan
    # BayesOptSearch with 256 daily growth samples after 1024 trials with max 250 grams.
    # Revision: This is just way too hard and unrealistic. Lettuce will also grow at 10 degrees.
    # Grow also at low degrees. Until 0
    "growth_temp_range": 25,
    # 95% of growth at 1200 ppm, 63% of growth at 400 ppm
    "growth_carbon_scalar": 400,
    # 70% of growth under lamps, typical daylight peak 95% growth
    "growth_par_scalar": 150,
    # PREV
    # 'growth_temp_range': 19.060141348751763,
    # 'growth_carbon_scalar': 1214.361207833735,
    # 'growth_par_scalar': 220.67842974727222,
    # Optimization_of_Light-Dark_Cycles_of_Lactuca_sativ
    # https://www.researchgate.net/publication/316429356_Optimization_of_Light-Dark_Cycles_of_Lactuca_sativa_L_in_Plant_Factory
    # region of L/D2 and T15-T22 seems to be the optimal
    # light-dark cycles for leaf lettuce (Lactuca sativa L. cv.
    # Greenwave) under PPF 245 mol m² s
    # https://hcs.osu.edu/sites/hcs/files/imce/images/PlantGrowthRequirements.pdf
    # 12 - 20 hours needed
    "min_hours_needed_for_max_growth": 10,
    # Business case for large scale crop
    # production in greenhouse facilities in
    # Iceland for the global market
    # Lettuce is a fast growing crop. With the right amount of light and optimum temperature (24 °C), it is possible to
    # grow a marketable size lettuce (330 g) in around 28 days.
    # 330/ 2.26   -> **1/28 -> 1.194
    "max_daily_growth": 1.194,
    # === Plant Quality Loss ====
    # Plant Death
    # Tolerance_of_Ten_Lettuce_Cultivars_to_High_Temperatures.pdf
    # Most lettuce cultivars were not able to germinate at temperatures over 35 degrees
    # At 30 degrees most lettuce cultivars were able to germinate which rapidly declined to zero cultivars at 35 deg.
    "temperature_quality_loss_rate": 0.01,
    "temperature_tolerance_upper": 30,
    "temperature_tolerance_lower": 0,
    # Tipburn of lettuce file:///home/woutervdb/Documents/papers_new/TipburnofLettuce.pdf
    # Rapid growth rate causes lettuce tipburn due to calcium deficiency
    # Partially made up:
    "tipburn_tolerance_growing_rate_upper": 0.80,
    "tipburn_quality_loss_rate": 0.01,
    # Extreme humidity - Drying out and fungus / disease growing.
    "humidity_tolerance_upper": 0.90,
    "humidity_tolerance_lower": 0.10,
    "humidity_quality_loss_rate": 0.001,
    # CO2 Exchange of the plant
    "mass_gain_to_co2_absorption_rate": 0.00001,  # [kg/kg]
    "respiration_rate": 1e-9,  # (kg/s)
    "absorption_to_respiration_rate": 0.5,
    # === Planting Density Model ===
    # Planting density sections in plants per m2.
    "density_steps": [80, 50, 25, 14],
    # Optimizing-the-planting-density-of-Lettuce-Lactuca-sativa-with-Tilapia-Oreochromis-niloticus-in-a-recirculation-aquaponic-system--1.pdf
    # Fixed density that overrides the density steps above. Set to None to prevent overriding
    "fixed_density_override_for_cost_model": 25,
}