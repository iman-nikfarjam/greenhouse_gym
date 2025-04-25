import unittest

from typing import Dict, Type

import numpy as np

from rl_greenhouse.agents.rule_based.rule_based_agent import RuleBased
from rl_greenhouse.greenhouse.types import Action
from rl_greenhouse.greenhouse import Greenhouse, GreenhouseNumpy
from rl_greenhouse.greenhouse import reward_functions
from rl_greenhouse.greenhouse.reward_functions import RewardFunction
from rl_greenhouse.utils.evaluation import play_actions_on_env

AgentType = RuleBased


def create_agents_to_test(
        greenhouse_type: type,
        reward_functions_to_test: Dict[str, Type[RewardFunction]]
) -> Dict[str, RuleBased]:
    functions: Dict[str, Greenhouse] = {
        reward_func_name: greenhouse_type(
            {"reward_function": reward_func, "growing_cycle": 7, "start_date_randomization": None}
        )
        for reward_func_name, reward_func in reward_functions_to_test.items()
    }
    agents = {reward_func_name: RuleBased(greenhouse) for reward_func_name, greenhouse in functions.items()}
    return agents


def create_example_actions():
    action_a = Action(2 / 3, 2 / 3, 2 / 3, 0.5, 0.5, 1)
    action_b = Action(1, 1, 1, 1, 1, 1)
    action_c = Action(0, 0, 0, 0, 0, 0)

    normalized_720_actions_numpy = [action_a.to_numpy()] * 240 + [action_b.to_numpy()] * 240 + [
        action_c.to_numpy()] * 240
    non_normalized_720_actions = [action_a.denormalize()] * 240 + [action_b.denormalize()] * 240 + [
        action_c.denormalize()] * 240
    return non_normalized_720_actions, normalized_720_actions_numpy


def create_greenhouses_and_action_sets():
    env_config = {"reward_function": reward_functions.BalanceReward, "start_date_randomization": None}
    greenhouse = Greenhouse(env_config)
    greenhouse_np = GreenhouseNumpy(env_config)

    non_normalized_720_actions, normalized_720_actions_numpy = create_example_actions()
    return ["Greenhouse", "Greenhouse Numpy"], \
           [greenhouse, greenhouse_np], \
           [non_normalized_720_actions, normalized_720_actions_numpy]


def generate_greenhouse_traces():
    names = []
    all_rewards = []
    sequences = []
    for greenhouse_name, greenhouse, actions in zip(*create_greenhouses_and_action_sets()):
        sequence, rewards, _ = play_actions_on_env(greenhouse, actions)
        names.append(greenhouse_name)
        sequences.append(sequence)
        all_rewards.append(rewards)

    return sequences, all_rewards, names


class GreenhouseResetsAndRewards(unittest.TestCase):
    reward_functions_to_test = {
        "MassReward": reward_functions.MassReward,
        "FinalBalanceReward": reward_functions.FinalBalanceReward,
        "DeltaMassReward": reward_functions.DeltaMassReward,
        "MassCreditedReward": reward_functions.GrowthReward,
        "QualityLossReward": reward_functions.QualityLossReward,
        "CostsReward": reward_functions.CostsReward,
        "BalanceCreditedReward": reward_functions.BalanceReward,
    }

    greenhouse_types = [Greenhouse, GreenhouseNumpy]

    def test_reward_sequential_consistency(self):
        """
        Tests if rewards of sequential runs in the same environment and conditions lead to identical rewards.
        """
        for greenhouse_type in self.greenhouse_types:
            agents = create_agents_to_test(greenhouse_type, self.reward_functions_to_test)

            for reward_func, agent in agents.items():
                rewards = []
                for run in range(3):
                    _, total_reward, _ = agent.step()
                    rewards.append(total_reward)
                first_reward = rewards[0]
                for reward in rewards:
                    self.assertEqual(
                        first_reward,
                        reward,
                        f"Running a sequential run on the basic environment with reward function {reward_func}.\n"
                        f"does not yield identical cumulative rewards after resetting.\n"
                        f"3 runs return rewards {rewards} while they should be identical."
                    )
                # print(reward_func, agent.env.name, rewards)

    def test_observation_sequential_consistency(self):
        """
        Tests if sequential runs of the same greenhouse return identical observations.
        """

        for greenhouse_name, greenhouse, actions in zip(*create_greenhouses_and_action_sets()):
            agent = RuleBased(greenhouse)
            sequences = []
            runs = []
            for run in range(3):
                sequence, _, _ = agent.step()
                sequences.append(sequence)
                runs.append(str(run))

            first_sequence = sequences[0]
            first_name = runs[0]
            for sequence, name in zip(sequences[1:], runs[1:]):
                self.assertEqual(
                    len(first_sequence),
                    len(sequence),
                    f"Sequence length unequal in identical conditions for same env in sequential runs. "
                    f"Run {first_name} has length {len(first_sequence)} while Run {name} has length {len(sequence)}."
                )

                for first_obs, obs in zip(first_sequence.observations, sequence.observations):
                    self.assertTrue(
                        np.allclose(first_obs.to_numpy(), obs.to_numpy(), rtol=0.001, atol=0.001),
                        f"Observations unequal in identical conditions for same env in sequential runs.\n"
                        f"Mismatch between Run {name} and Run {name}.\n"
                        f"First obs:\n{first_obs}\n\nSecond obs:\n{obs}\n"
                    )

    def test_reward_different_greenhouses_consistency(self):
        """
        Tests if all greenhouse types under the same conditions return the same results.
        """
        _, all_rewards, names = generate_greenhouse_traces()
        first_rewards = all_rewards[0]
        first_name = names[0]
        for rewards, name in zip([all_rewards[0], all_rewards[1]], [names[0], names[1]]):
            self.assertEqual(
                len(first_rewards),
                len(rewards),
                f"Reward length unequal in identical conditions for different envs. "
                f"Env {first_name} has length {len(first_rewards)} while Env {name} has length {len(rewards)}."
            )

            for step_nr, (first_reward, reward) in enumerate(zip(first_rewards, rewards)):
                self.assertAlmostEqual(
                    first_reward,
                    reward,
                    places=3,
                    msg=
                    f"Rewards for reward function BalanceCreditedReward unequal for Env {first_name} and Env {name}.\n"
                    f"At step {step_nr} reward for Env {first_name} is {first_reward} and for Env {name} it is {reward}"
                )

    def test_observation_different_greenhouses_consistency(self):
        """
        Tests if all greenhouse types under the same conditions return the same results.
        """
        sequences, _, names = generate_greenhouse_traces()
        first_sequence = sequences[0]
        first_name = names[0]
        for sequence, name in zip(sequences[1:], names[1:]):
            self.assertEqual(
                len(first_sequence),
                len(sequence),
                f"Sequence length unequal in identical conditions for different envs. "
                f"Env {first_name} has length {len(first_sequence)} while Env {name} has length {len(sequence)}."
            )

            for first_obs, obs in zip(first_sequence.observations[1:], sequence.observations[1:]):
                self.assertTrue(
                    np.allclose(first_obs.to_numpy(), obs.to_numpy(), rtol=0.001, atol=0.001),
                    f"Observations unequal in identical conditions for different envs.\n"
                    f"Mismatch between Env {name} and Env {name}.\n"
                    f"First obs:\n{first_obs}\n\nSecond obs:\n{obs}\n"
                )
