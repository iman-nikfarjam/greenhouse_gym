from typing import Dict, Optional

import numpy as np
try:  # pragma: no cover - optional dependency
    from ray.rllib.agents.callbacks import DefaultCallbacks  # type: ignore
    from ray.rllib.env import BaseEnv  # type: ignore
    from ray.rllib.evaluation import RolloutWorker, MultiAgentEpisode  # type: ignore
    from ray.rllib.policy import Policy  # type: ignore
    from ray.rllib.utils.typing import PolicyID  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class DefaultCallbacks:  # type: ignore
        """Fallback stub when Ray is not installed."""
        pass

    class BaseEnv:  # type: ignore
        pass

    class RolloutWorker:  # type: ignore
        pass

    class MultiAgentEpisode:  # type: ignore
        pass

    class Policy:  # type: ignore
        pass

    PolicyID = str  # type: ignore

from rl_greenhouse.greenhouse.types import Observation, Action


class GreenhouseInformation(DefaultCallbacks):
    FINAL_STEP_TRACKABLES = [
        "time",
        "mass",
        "quality",
        "plants_per_m2",
        "average_head_per_m2",
        "gains_plant",
        "cum_electricity_usage",
        "cum_heat_usage",
        "cum_carbon_usage",
        "costs_var_electricity",
        "costs_var_heat",
        "costs_var_carbon",
        "costs_var_total",
        "costs_fix_total",
        "balance",
        "quality",
    ]
    PERCENTILES = [5, 50, 95]
    QUANTILE_TRACKABLES_OBS = ["temperature_air", "par_light", "carbon_ppm"]
    QUANTILE_TRACKABLES_ACTIONS = [
        "heater_power",
        "ventilation_position",
        "co2_flow_rate",
        "screen_blackout_position",
        "screen_transparent_position",
        "enable_lamps",
    ]

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:

        for metric in self.QUANTILE_TRACKABLES_OBS + self.QUANTILE_TRACKABLES_ACTIONS:
            episode.user_data[metric] = []

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:

        if len(policies) > 1:
            extra_args = ["protagonist"]
        else:
            extra_args = []

        # Info contains the Non-normalized action and obs directly from the Greenhouse Class.
        last_info = episode.last_info_for(*extra_args)
        last_obs: Observation = last_info["obs"]
        for attribute in self.QUANTILE_TRACKABLES_OBS:
            episode.user_data[attribute].append(last_obs.get(attribute))

        last_action: Action = last_info["action"]
        for attribute in self.QUANTILE_TRACKABLES_ACTIONS:
            episode.user_data[attribute].append(getattr(last_action, attribute))

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        if len(policies) > 1:
            extra_args = ["protagonist"]
        else:
            extra_args = []

        final_obs = episode.last_info_for(*extra_args)["obs"]

        for attribute in self.FINAL_STEP_TRACKABLES:
            episode.custom_metrics[f"final_{attribute}"] = final_obs.get(attribute)

        for attribute in self.QUANTILE_TRACKABLES_OBS + self.QUANTILE_TRACKABLES_ACTIONS:
            prefix = "action_" if attribute in self.QUANTILE_TRACKABLES_ACTIONS else ""
            percentile_values = np.percentile(np.array(episode.user_data[attribute]), self.PERCENTILES)
            for percentile, value in zip(self.PERCENTILES, percentile_values):
                episode.custom_metrics[f"{attribute}_{prefix}{percentile}_quantile"] = float(value)
