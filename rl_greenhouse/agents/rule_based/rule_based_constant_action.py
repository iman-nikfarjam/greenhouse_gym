"""
This agent is based on heuristic growing information mainly provided by research at Cornell University.
A link to the research can be found here: https://cea.cals.cornell.edu/crops/.

It does not inherit from any RAY trainer as it is not meant to be run in parallel.
"""
from rl_greenhouse.greenhouse.types import Action, Sequence
from rl_greenhouse.greenhouse import Greenhouse


class RuleBasedConstant:
    """
    An agent that uses constant set-points as an action.
    """

    NORMALIZED_ACTION = Action(
        heater_power=0.25,
        ventilation_position=0.1,
        co2_flow_rate=1,
        screen_blackout_position=0,
        screen_transparent_position=0,
        enable_lamps=0,
    )

    def __init__(self, env: Greenhouse, action=None, normalized_action=None):
        self.env = env
        if action and normalized_action:
            raise AssertionError("Cannot set both action and normalized action.")
        elif action:
            self.action = action
        elif normalized_action:
            self.action = normalized_action.denormalize()
        else:
            self.action = self.NORMALIZED_ACTION.denormalize()

        # print(self.action)

    def step(self) -> (Sequence, float, float):
        """
        Plays one episode in the Greenhouse Simulator
        """
        done = False
        all_obs = [self.env.reset()]
        all_actions = []
        total_reward = 0
        while not done:
            action = self.action
            all_actions.append(action)
            obs, reward, done, info = self.env.step(action)
            all_obs.append(obs)
            total_reward += reward

        return Sequence(all_obs, all_actions), total_reward, info['final_balance']
