import os
import pickle
from pathlib import Path
from typing import Type, Union

import numpy as np
from gym.spaces import Box
try:  # pragma: no cover - optional dependency
    from ray.rllib.agents import Trainer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class Trainer:  # type: ignore
        """Fallback stub when Ray is not installed."""
        pass


def restore_config(checkpoint_path: str) -> dict:
    """Restores a config file based on a checkpoint path"""
    params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path))), "params.pkl")
    with open(params_path, 'rb') as f:
        config = pickle.load(f)
    return config


def restore_agent(trainer_type: Type[Trainer], checkpoint_path: Union[str, Path], explore: bool = True,
                  single_worker: bool = False, num_gpus=0) -> (Trainer, dict):
    """
    Restores an agent and its config file from a checkpoint.

    :param trainer_type: The type of the trainer
    :param checkpoint_path: The checkpoint path name.
    :param explore: False to disable exploration. Defaults to True, which overrides nothing in the config file.
    :param single_worker: True to override config with 1 worker and no gpu usage / evaluation
    """
    config = restore_config(checkpoint_path)
    config['action_space'] = Box(0, 1, shape=(6,), dtype=np.float32)

    if not explore:
        config["explore"] = False
        config["evaluation_config"]["explore"] = False
        config['in_evaluation']: True

    config["num_workers"] = 1
    config["num_envs_per_worker"] = 1
    config["num_gpus"] = num_gpus
    config["evaluation_num_workers"] = 0
    config["evaluation_interval"] = 0
    config["evaluation_num_episodes"] = 1
    config["evaluation_parallel_to_training"] = False
    # config["evaluation_duration"] = "auto"

    # config['env_config']['roof_heat_transfer_rate'] = 2
    # config['evaluation_config']['env_config']['roof_heat_transfer_rate'] = 2

    trainer = trainer_type(config)
    trainer.restore(checkpoint_path)
    return trainer, config["env_config"]


def restore_agent_policy(trainer_type: Type[Trainer], checkpoint_path: str, num_gpus: int = None):
    config = restore_config(checkpoint_path)
    config["num_workers"] = 1
    config["num_envs_per_worker"] = 1

    if num_gpus:
        config['num_gpus'] = num_gpus

    trainer = trainer_type(config)
    trainer.restore(checkpoint_path)
    return trainer.get_policy()


def restore_with_env_overrides(trainer_type: Type[Trainer], checkpoint_path: str, env_config_overrides: dict):
    config = restore_config(checkpoint_path)
    config["num_workers"] = 1
    config["num_envs_per_worker"] = 1

    env_config = config['env_config']
    env_config = {**env_config, **env_config_overrides}
    config['env_config'] = env_config

    env_config = config['evaluation_config']['env_config']
    env_config = {**env_config, **env_config_overrides}
    config['evaluation_config']['env_config'] = env_config

    trainer = trainer_type(config)
    trainer.restore(checkpoint_path)
    return trainer
