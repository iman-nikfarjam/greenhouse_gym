"""
PPO Training Script. It creates a configuration file as uses Ray Tune to start training.
"""
import logging
from datetime import datetime
from pathlib import Path
from ray import tune, init

from rl_greenhouse.utils.ray_trainer_config_building import create_complete_config


if __name__ == "__main__":

    algorithm = "PPO"
    experiment_name = "OctoberPretrainedEnergyCrisisAgent"

    ray_results_directory_path = Path(__file__).parent.parent.parent.parent / "ray_results"
    ray_results_directory_path.mkdir(parents=True, exist_ok=True)
    ray_results_directory = str(ray_results_directory_path)

    env_config_overrides = {
        "price_electricity_on_peak": 0.58,  # EUR / kWh
        "price_electricity_off_peak": 0.58,  # EUR / kWh
        "price_heating": 0.044,  # EUR / MJ (= 158.59 EUR / MWh)
        "start_date": datetime(year=2001, month=10, day=1)
    }

    config_path = Path("ppo.yaml")
    config = create_complete_config(config_path=config_path, env_config_overrides=env_config_overrides)
    config["num_envs_per_worker"] = 1
    config["num_workers"] = 14  # We reserve 1 core for the scheduler process and one for the evaluation process.

    init(logging_level=logging.INFO, num_gpus=0, num_cpus=16, local_mode=False)

    tune.run(
        algorithm,
        local_dir=ray_results_directory,
        stop={"timesteps_total": 10_000_000},
        num_samples=1,
        name=experiment_name,
        config=config,
        checkpoint_freq=20,
        checkpoint_at_end=True,
    )

# restore=os.path.join(
#     RAY_RESULTS_FOLDER,
#     "1_ppo_final/PPO_WrappedGreenhouse_f2147_00000_0_2022-02-09_14-55-32/checkpoint_000020/checkpoint-20"
# )
