from copy import deepcopy
from inspect import isfunction
from pathlib import Path
from typing import Type, Union

from yaml import full_load

from rl_greenhouse.greenhouse import GreenhouseNumpy, WrappedGreenhouse
from rl_greenhouse.greenhouse.custom_callbacks import GreenhouseInformation
from rl_greenhouse.greenhouse.env_wrappers import WrappedGreenhouseMultiAgent

EnvType = Union[Type[GreenhouseNumpy], Type[WrappedGreenhouse]]


def create_complete_config(
        config_path: Path, env_config_overrides: dict = None
) -> dict:
    with open(config_path, "r") as stream:
        default_config = full_load(stream)

    if "env_config" in default_config.keys() and len(default_config["env_config"].get("wrappers", [])) > 0:
        if "multiagent" in default_config.keys() or \
                "AdversarialWrapper" in default_config["env_config"].get("wrappers", []):
            env = WrappedGreenhouseMultiAgent
        else:
            env = WrappedGreenhouse
    else:
        env = GreenhouseNumpy

    config = {
        **default_config,
        "env": env,
        # "observation_space": GreenhouseNumpy.observation_space,
        "callbacks": GreenhouseInformation,
    }

    env_config = config.get("env_config", {})
    if env_config_overrides is not None:
        env_config = {**env_config, **env_config_overrides}

    config["env_config"] = deepcopy(env_config)
    config["env_config"]["wrappers"] = config["env_config"].get("wrappers", [])

    config["evaluation_config"] = {}
    config["evaluation_config"]["explore"] = False
    config["evaluation_config"]["env_config"] = {}  # deepcopy(env_config)
    config["evaluation_config"]["env_config"]["start_date_randomization"] = "benchmark"
    config["evaluation_config"]["env_config"]["start_date_decade"] = 1

    return config


def create_experiment_name(env: EnvType, experiment_name: str, config_greenhouse: str, config_algorithm: str,
                           default_config: str) -> str:
    """

    :param env: The environment type to use.
    :param experiment_name: Name for the experiment.
    :param config_greenhouse:
    :param config_algorithm:
    :param default_config:
    :return:
    """
    if isfunction(env):
        name = "wrapper"
    else:
        name = env.name
    return f"{experiment_name}_{name}_{config_greenhouse}_{default_config}_{config_algorithm}"


def combine_configs(
        env: Type[GreenhouseNumpy],
        algorithm_config: dict,
        greenhouse_config: dict,
        default_config: dict,
):
    """
    Combines configs into a RAY compatible config.

    :param env: The environment type to use.
    :param algorithm_config:
    :param greenhouse_config:
    :param default_config:
    """
    env_config = deepcopy(greenhouse_config)
    eval_env_config = deepcopy(env_config)
    eval_env_config["start_date_randomization"] = "benchmark"
    eval_env_config["start_date_decade"] = 1

    config = {
        **algorithm_config,
        "env": env,
        "env_config": env_config,  # Env config regular
        "evaluation_config": {
            **algorithm_config,
            "explore": False,
            "env_config": eval_env_config,  # Env config evaluation!!
        },
        # "observation_space": GreenhouseNumpy.observation_space,
        "callbacks": GreenhouseInformation,
    }

    full_config = deepcopy(default_config)
    full_config.update(config)
    return full_config
