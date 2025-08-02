import logging
import math
from abc import ABC, abstractmethod

from rl_greenhouse.greenhouse.types import Observation


class RewardFunction(ABC):
    """
    Reward function base class defining the interface of a reward function.
    """

    def __init__(self, config: dict):
        ...

    def reset(self):
        """Resets the reward function."""

    @abstractmethod
    def calculate(self, full_state: Observation, done: bool) -> float:
        """Calculates the reward"""


class MassReward(RewardFunction):
    """Rewards the user on plant mass."""

    def __init__(self, config: dict):
        super().__init__(config)

    def calculate(self, full_state: Observation, done: bool) -> float:
        return full_state.plant.mass


class DeltaMassReward(RewardFunction):
    def __init__(self, config: dict):
        self.shoot_mass = config["shoot_mass"]
        self.mass = self.shoot_mass
        super().__init__(config)

    def reset(self):
        self.mass = self.shoot_mass

    def calculate(self, full_state: Observation, done: bool) -> float:
        gained = max(full_state.plant.mass - self.mass, 0)
        self.mass += gained
        return gained


class FinalBalanceReward(RewardFunction):
    """Rewards only for the final balance, like a grower is rewarded in real life."""

    def calculate(self, full_state: Observation, done: bool) -> float:
        if done:
            return full_state.info.balance
        return 0


class GrowthReward(RewardFunction):
    """
    Credits the algorithm for mass gains and rectified for small mass gains at the start of the growth cycle.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.shoot_mass = config["shoot_mass"]
        self.final_mass = 250
        self.mass = self.shoot_mass
        self.growth_required = self.final_mass / self.shoot_mass
        if config["fixed_density_override_for_cost_model"] is None:
            logging.warning("fixed_density_override_for_cost_model is None! Assuming a final density of 27")
            self.final_gains = 27 * config["max_price_plant"]
        else:
            self.final_gains = config["fixed_density_override_for_cost_model"] * config["max_price_plant"]

    def reset(self):
        self.mass = self.shoot_mass

    def calculate(self, full_state: Observation, done: bool) -> float:
        growth_current = full_state.plant.mass / self.mass
        if growth_current == 1:
            return 0
        hours_required = math.log(self.growth_required, growth_current)  # X, base
        self.mass = full_state.plant.mass

        # We are not sure where in the hour to end, so we give the average expected reward for the terminal step.
        if done:
            return 0.5 * self.final_gains / hours_required

        return self.final_gains / hours_required


class QualityLossReward(RewardFunction):
    """ """

    def __init__(self, config: dict):
        super().__init__(config)
        self.quality = 1
        if config["fixed_density_override_for_cost_model"] is None:
            logging.warning("fixed_density_override_for_cost_model is None! Assuming a final density of 27")
            self.final_gains = 27 * config["max_price_plant"]
        else:
            self.final_gains = config["fixed_density_override_for_cost_model"] * config["max_price_plant"]

    def reset(self):
        self.quality = 1

    def calculate(self, full_state: Observation, done: bool) -> float:
        quality_decrease = self.quality - full_state.plant.quality
        self.quality = full_state.plant.quality
        return -quality_decrease * self.final_gains


class CostsReward(RewardFunction):
    def __init__(self, config: dict):
        super().__init__(config)
        self.total_costs = 0

    def reset(self):
        self.total_costs = 0

    def calculate(self, full_state: Observation, done: bool) -> float:
        total_cost = full_state.info.costs_fix_total + full_state.info.costs_var_total
        cost_increase = total_cost - self.total_costs
        self.total_costs = total_cost
        return -cost_increase


class GoalReachingMotivator(RewardFunction):
    """
    Penalize the agent for not reaching the goal in time.

    Almost reaching the goal means not being able to reach it for sure just before the end.
    If we know it very early, we are performing very badly.

    Once it is detected that the agent cannot make it:
        - Define days remaining d
        - Determine if under maximum growth we can still reach 250 grams
        - If not:
            Apply a penalty during each time step.
    """

    PUNISHMENT_REWARD = -0.01

    def __init__(self, config: dict):
        super().__init__(config)
        self.final_mass = 250
        self.max_daily_growth = config["max_daily_growth"]
        self.max_days = config["growing_cycle"]
        self.cannot_make_it = False

    def reset(self):
        self.cannot_make_it = False

    def calculate(self, full_state: Observation, done: bool) -> float:
        if self.cannot_make_it:
            return self.PUNISHMENT_REWARD

        if full_state.plant.mass >= 250:
            return 0

        # We floor it, as this gives the agent the advantage of the doubt. (Today it can still grow for a full day!)
        # It could be calculated exactly though, but I am kinda lazy.
        days_left = self.max_days - math.floor(full_state.info.time / 24)

        maximum_achievable_growth = self.max_daily_growth ** days_left
        growth_required = self.final_mass / full_state.plant.mass

        if growth_required > maximum_achievable_growth:
            self.cannot_make_it = True
            return self.PUNISHMENT_REWARD

        return 0


class MissedProfitsMotivator(RewardFunction):
    """
    Penalize the agent for each timestep. Faster growing means more cycles per year.

    Expected profit per m2 per year is 15 EUR / m2

    Each day of delaying a growing next growing cycle costs money.

    We apply -15/365/24 punishment at each step.
    """

    PUNISHMENT_REWARD = -15 / 365 / 24

    def reset(self):
        pass

    def calculate(self, full_state: Observation, done: bool) -> float:
        return self.PUNISHMENT_REWARD


class NotReachingGoalPenalty(RewardFunction):
    """
    Penalize the agent for not reaching the goal in time.

    Once it is detected that the agent cannot make it:
        - Define days remaining d
        - Calculate Current Growth Reward Awarded
        - Apply the following Penalty for each step:
            reward = -growth_reward_awarded / n_hours remaining - growth_reward at this step
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.final_mass = 250
        self.max_daily_growth = config["max_daily_growth"]
        self.max_days = config["growing_cycle"]

        self.cannot_make_it = False
        self.punishment_reward = None
        self.quality_reward_emitted = 0
        self.growth_reward_emitted = 0
        self.growth_reward_function = GrowthReward(config)
        self.quality_reward_function = QualityLossReward(config)

    def reset(self):
        self.cannot_make_it = False
        self.quality_reward_emitted = 0
        self.growth_reward_emitted = 0
        self.punishment_reward = None
        self.quality_reward_function.reset()
        self.growth_reward_function.reset()

    def _calculate_punishment_reward(self, full_state: Observation, done: bool) -> float:
        quality_and_growth_reward = self.growth_reward_function.calculate(
            full_state, done
        ) + self.quality_reward_function.calculate(full_state, done)
        reward = self.punishment_reward - quality_and_growth_reward
        return reward

    def calculate(self, full_state: Observation, done: bool) -> float:
        if self.cannot_make_it:
            return self._calculate_punishment_reward(full_state, done)

        if full_state.plant.mass >= 250:
            return 0

        # We floor it, as this gives the agent the advantage of the doubt. (Today it can still grow for a full day!)
        # It could be calculated exactly though, but I am kinda lazy.
        days_left = self.max_days - math.floor(full_state.info.time / 24)
        hours_left = self.max_days * 24 - full_state.info.time

        maximum_achievable_growth = self.max_daily_growth ** days_left
        growth_required = self.final_mass / full_state.plant.mass

        if growth_required > maximum_achievable_growth:
            self.cannot_make_it = True
            self.punishment_reward = -(self.quality_reward_emitted + self.growth_reward_emitted) / hours_left
            return self._calculate_punishment_reward(full_state, done)

        self.quality_reward_emitted += self.quality_reward_function.calculate(full_state, done)
        self.growth_reward_emitted += self.growth_reward_function.calculate(full_state, done)
        return 0


class BalanceReward(RewardFunction):
    """Reward based directly on changes in greenhouse balance."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.prev_balance = 0.0

    def reset(self):
        self.prev_balance = 0.0

    def calculate(self, full_state: Observation, done: bool) -> float:
        current_balance = full_state.info.balance
        reward = current_balance - self.prev_balance
        self.prev_balance = current_balance
        return reward


class DailyBalanceReward(RewardFunction):
    """
    Credits the algorithm for profit / loss when applicable. Batches the reward into a

    Ensures cumulative reward equals the final profit, even in case of an early finish.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.reward_today = 0
        self.reward_func = BalanceReward(config)

    def reset(self):
        self.reward_today = 0
        self.reward_func.reset()

    def calculate(self, full_state: Observation, done: bool) -> float:
        reward = self.reward_func.calculate(full_state, done)
        self.reward_today += reward

        if full_state.info.hour == 0 or done:
            accumulated_reward = self.reward_today
            self.reward_today = 0
            return accumulated_reward

        return 0
