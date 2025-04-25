import unittest

from rl_greenhouse.agents.rule_based.rule_based_agent import RuleBased
from rl_greenhouse.greenhouse.env import GreenhouseNumpy
from rl_greenhouse.utils.evaluation import benchmark_agent


class GreenhouseDomainRandomization(unittest.TestCase):
    """
    Tests the different configuration options for domain randomization in greenhouses.

    Expected behaviour:

        When 'domain_randomization_ranges' is not None:
            Error on:
                It should a dict, otherwise error
                If keys do not correspond with greenhouse config keys throw an error

            A greenhouse reset should result in a greenhouse with randomized config parameters as specified in the dict

            E.g.
            {
                'roof_h': (0.9, 1.1)
            }
            should result in a uniform sampled greenhouse instance with a roof enthalpy value of 0.9 - 1.1.

        When 'domain_randomization' is None:
            Greenhouse should use the values specified in the config as normal.
    """
    DETERMINISTIC_CONFIG = {"start_date_randomization": None}

    BENCHMARK_CONFIG = {"start_date_randomization": 'benchmark'}

    DOMAIN_RANDOMIZATION_CONFIG = {
        "domain_randomization_ranges": {
            "reflectance_greenhouse": (0.3, 0.5),
            "reflectance_blackout": (0.8, 1.0),
            "reflectance_transparent": (0.2, 0.4),
            "fraction_sun_heat_absorbed": (0.2, 0.4),
            "heat_blocked_by_screens": (0.6, 0.8),
        }
    }

    def test_error_on_invalid_config_type(self):
        """Throw an error if 'domain_randomization_ranges' is not None or Dict."""
        config = {"domain_randomization_ranges": 9, **self.DETERMINISTIC_CONFIG}
        try:
            benchmark_agent(RuleBased(GreenhouseNumpy(config)), config, standard_bench=False,
                            greenhouse_type=GreenhouseNumpy)
            self.fail("Invalid types type for parameter 'domain_randomization_ranges' "
                      "in the greenhouse config did not throw an error!")
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError))

    def test_error_on_invalid_key(self):
        """Throw an error if a key is not contained in the default greenhouse dict."""
        config = {"domain_randomization_ranges": {"weird_ass_key": 1}, **self.DETERMINISTIC_CONFIG}
        try:
            benchmark_agent(RuleBased(GreenhouseNumpy(config)), config, standard_bench=False,
                            greenhouse_type=GreenhouseNumpy)
            self.fail("Invalid key type for parameter inside 'domain_randomization_ranges' "
                      "in the greenhouse config did not throw an error!")
        except Exception as e:
            self.assertTrue(isinstance(e, KeyError))

    def test_first_run_already_randomized(self):
        """Tests if the first run is already randomized as it should be"""
        config = {**self.DOMAIN_RANDOMIZATION_CONFIG, **self.DETERMINISTIC_CONFIG}
        greenhouse = GreenhouseNumpy(config)
        self.assertTrue(
            isinstance(greenhouse.current_domain_randomized_config, dict),
            "Domain Randomization is not applied on the first initialization of the greenhouse"
        )

    def test_first_run_no_randomization_on_benchmark_mode(self):
        """The first run should not use randomization in benchmark mode."""
        config = {**self.DOMAIN_RANDOMIZATION_CONFIG, **self.BENCHMARK_CONFIG}
        greenhouse = GreenhouseNumpy(config)
        self.assertFalse(
            isinstance(greenhouse.current_domain_randomized_config, dict),
            "Domain Randomization is incorrectly applied on the first initialization of the greenhouse "
            "when benchmark mode is enabled"
        )

    def test_no_randomization_on_benchmark_mode(self):
        """Tests if results are deterministic in benchmark mode."""
        config = {**self.DOMAIN_RANDOMIZATION_CONFIG, **self.BENCHMARK_CONFIG}
        balance_run_1 = benchmark_agent(RuleBased(GreenhouseNumpy(config)), config, standard_bench=True,
                                        greenhouse_type=GreenhouseNumpy)
        balance_run_2 = benchmark_agent(RuleBased(GreenhouseNumpy(config)), config, standard_bench=True,
                                        greenhouse_type=GreenhouseNumpy)
        self.assertEqual = (
            balance_run_1, balance_run_2,
            "Results for two sequential benchmark runs should be equal even in case of domain randomization!"
        )

    def test_different_greenhouses_should_yield_different_results(self):
        """Tests if multiple runs with domain randomized greenhouses yield different results."""
        config = {**self.DOMAIN_RANDOMIZATION_CONFIG, **self.DETERMINISTIC_CONFIG}
        balance_run_1 = benchmark_agent(RuleBased(GreenhouseNumpy(config)), config, standard_bench=False,
                                        greenhouse_type=GreenhouseNumpy)
        balance_run_2 = benchmark_agent(RuleBased(GreenhouseNumpy(config)), config, standard_bench=False,
                                        greenhouse_type=GreenhouseNumpy)
        self.assertNotEqual = (
            balance_run_1, balance_run_2,
            "Results for two sequential benchmark runs should be equal even in case of domain randomization!"
        )

    def test_sequential_resets_yield_different_greenhouses(self):
        config = {**self.DOMAIN_RANDOMIZATION_CONFIG, **self.DETERMINISTIC_CONFIG}
        greenhouse = GreenhouseNumpy(config)

        reflectance_greenhouse_values = []
        for i in range(10):
            greenhouse.reset()
            reflectance_greenhouse_values.append(greenhouse.greenhouse_model_components[2].reflectance_greenhouse)

        self.assertTrue(
            len(set(reflectance_greenhouse_values)) > 8,
            "Greenhouse values in sequential resets with domain randomization are not randomized!"
        )

    def test_randomized_greenhouses_are_within_domain_randomization_range(self):
        """Tests if all sampled greenhouses contain parameters within the given domain randomization range."""
        config = {**self.DOMAIN_RANDOMIZATION_CONFIG, **self.DETERMINISTIC_CONFIG}
        greenhouse = GreenhouseNumpy(config)

        for i in range(10):
            greenhouse.reset()
            current_config = greenhouse.current_domain_randomized_config
            for param, (low, high) in self.DOMAIN_RANDOMIZATION_CONFIG['domain_randomization_ranges'].items():
                self.assertTrue(
                    low <= current_config[param] <= high,
                    f"Value for {param} is {current_config[param]} "
                    f"is outside its domain randomization range of ({low}, {high})"
                )
