import unittest

from typing import Dict, Type

from rl_greenhouse.agents.rule_based.rule_based_agent import RuleBased
from rl_greenhouse.greenhouse import Greenhouse, GreenhouseNumpy
from rl_greenhouse.greenhouse import reward_functions
from rl_greenhouse.greenhouse.reward_functions import RewardFunction

AgentType = RuleBased

GROWING_CYCLE_LENGTH = 90


def create_agents_to_test(reward_functions_to_test: Dict[str, Type[RewardFunction]], ) -> Dict[str, RuleBased]:
    functions: Dict[str, Greenhouse] = {
        reward_func_name: GreenhouseNumpy({"reward_function": reward_func, 'growing_cycle': GROWING_CYCLE_LENGTH})
        for reward_func_name, reward_func in reward_functions_to_test.items()
    }
    agents = {reward_func_name: RuleBased(greenhouse) for reward_func_name, greenhouse in functions.items()}
    return agents


class GreenhouseResetsAndRewards(unittest.TestCase):
    reward_functions_to_test = {
        "FinalBalanceReward": reward_functions.FinalBalanceReward,
        "BalanceReward": reward_functions.BalanceReward,
        "DailyBalanceReward": reward_functions.DailyBalanceReward,
    }

    def test_reward_match_on_finished_cycle_final_balance(self):
        """
        The reward should match for a finished cycle.

        If the cycle is not finished we know that the reward is already inaccurate.
        This is one of the core assumptions of our reward function.
        """
        agents = create_agents_to_test(self.reward_functions_to_test)

        for reward_func, agent in agents.items():
            for run in range(5):
                seq, total_reward, final_balance = agent.step()
                if len(seq) >= GROWING_CYCLE_LENGTH * 24:
                    continue
                self.assertAlmostEqual(
                    total_reward,
                    final_balance,
                    delta=0.15,
                    msg=f"Final Balance does not equal cumulative reward for reward function {reward_func}"
                )
                print(total_reward, final_balance)
