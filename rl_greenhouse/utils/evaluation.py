import copy
from typing import List, Type, Union

import numpy as np
from ray.rllib.agents import Trainer

from rl_greenhouse.agents.rule_based.rule_based_agent import RuleBased
from rl_greenhouse.greenhouse.types import Sequence, Action
from rl_greenhouse.greenhouse import GreenhouseNumpy, WrappedGreenhouse, GREENHOUSE_TYPE, Greenhouse, WrappedGreenhouseMultiAgent

TrainerLike = Union[Trainer, RuleBased]


def play_actions_on_env(
        env: Union[GreenhouseNumpy, Greenhouse, WrappedGreenhouse, WrappedGreenhouseMultiAgent],
        all_actions: Union[List[Action], List[np.ndarray]],
        stop_when_done=False
) -> (Sequence, List[float], float):
    """
    Plays the actions on the given environment, returns a sequence.

    :param env:
    :param all_actions:
    :param stop_when_done:
    """
    if isinstance(env, (GreenhouseNumpy, WrappedGreenhouse, WrappedGreenhouseMultiAgent)):
        assert isinstance(all_actions[0], np.ndarray), \
            f"For env of type {type(env)} we need actions of type Action, not {type(all_actions[0])}"
        return _play_actions_on_env_numpy(env, all_actions, stop_when_done)
    elif isinstance(env, Greenhouse):
        assert isinstance(all_actions[0], Action), \
            f"For env of type Greenhouse we need actions of type Action, not {type(all_actions[0])}"
        return _play_actions_on_regular_env(env, all_actions, stop_when_done)
    raise TypeError(f"Greenhouse of type {type(env)} is not supported!")


def _play_actions_on_regular_env(
        env: Greenhouse,
        all_actions: List[Action],
        stop_when_done=False
) -> (Sequence, List[float], float):
    """
    Plays the actions on the given environment, returns a sequence.

    :param env:
    :param all_actions:
    :param stop_when_done:
    """
    time = 0
    observations = [env.reset()]
    rewards = []
    final_balance = 0
    for action in all_actions:
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        observations.append(obs)
        time += 1
        final_balance = info['final_balance']
        if done and stop_when_done:
            return Sequence(actions=all_actions[:len(observations) - 1], observations=observations), rewards
    return Sequence(actions=all_actions, observations=observations), rewards, final_balance


def _play_actions_on_env_numpy(
        env: Union[GreenhouseNumpy, WrappedGreenhouse],
        all_actions: List[np.array],
        stop_when_done=False
) -> (Sequence, List[float], float):
    """
    Plays the actions on the given environment, returns a sequence.

    :param env:
    :param all_actions:
    :param stop_when_done:
    """
    env.reset()
    all_rewards = []
    all_infos = []
    for action in all_actions:
        obs, reward, done, info = env.step(action)
        all_rewards.append(reward)
        all_infos.append(info)
        if done and stop_when_done:
            break
    sequence = Sequence(
        [all_infos[0]['obs']] + [info['obs'] for info in all_infos], [info['action'] for info in all_infos]
    )
    return sequence, all_rewards, all_infos[-1]["final_balance"]


def evaluate_agent_on_env_numpy(env: GREENHOUSE_TYPE, agent: TrainerLike) -> (Sequence, List[float], float):
    """
    Plays the actions on the given environment, returns a sequence.

    :param env:
    :param agent: Trainer object with compute_single_action implemented.
    """
    obs = env.reset()
    all_obs = []
    all_actions = []
    all_rewards = []
    all_infos = []
    final_balance = 0
    done = False
    while not done:
        action = agent.compute_single_action(obs, explore=False, unsquash_action=False)
        if not isinstance(agent, RuleBased):
            action = (action + 1) / 2

        obs, reward, done, info = env.step(action)
        all_obs.append(info["obs"])
        all_actions.append(info["action"])
        all_infos.append(info)
        final_balance = info["final_balance"]
        all_rewards.append(reward)

    all_obs = [all_obs[0]] + all_obs
    return Sequence(all_obs, all_actions, env_infos=all_infos), all_rewards, final_balance


def evaluate_rarl_agent_on_env_numpy(env: GREENHOUSE_TYPE, protagonist: TrainerLike, adversarial: TrainerLike) \
        -> (Sequence, List[float], float):
    """
    Plays the actions on the given environment, returns a sequence.

    :param env:
    :param protagonist: Trainer object with compute_single_action implemented.
    :param adversarial: Adversarial Agent
    """
    obs = env.reset()
    all_obs = []
    all_actions = []
    all_rewards = []
    final_balance = 0
    done = False
    while not done:
        action_p = protagonist.compute_single_action(obs, explore=False, unsquash_action=False)
        if not isinstance(protagonist, RuleBased):
            action_p = (action_p + 1) / 2
        action_a = adversarial.compute_single_action(obs, explore=False, unsquash_action=False)
        if not isinstance(adversarial, RuleBased):
            action_a = (action_a + 1) / 2

        obs, reward, all_done, info = env.step({'protagonist': action_p, 'adversarial': action_a})
        info = info['protagonist']
        all_obs.append(info["obs"])
        all_actions.append(info["action"])
        final_balance = info["final_balance"]
        done = all_done['protagonist']
        all_rewards.append(reward['protagonist'])

    all_obs = [all_obs[0]] + all_obs
    return Sequence(all_obs, all_actions), all_rewards, final_balance


def evaluate_agent(agent: Trainer) -> (float, dict):
    """
    Evaluates the agent

    Asserts exploration mode is False

    :return:
    """
    agent_config = agent.get_policy().config
    assert not agent_config['explore'], "Agent is on explore!"
    if 'evaluation_config' in agent_config.keys() and 'explore' in agent_config['evaluation_config'].keys():
        assert not agent_config['evaluation_config']['explore'], "Agent is on explore in the evaluation mode!"

    eval_dict = agent.evaluate()
    eval_final_balance = eval_dict['evaluation']['episode_reward_mean']
    return eval_final_balance, eval_dict


def benchmark_agent(
        agent: TrainerLike,
        env_config: Union[List[dict], dict],
        standard_bench: bool = True,
        benchmark_decade: int = 1,
        greenhouse_type: Type[GreenhouseNumpy] = WrappedGreenhouse
) -> float:
    """
    Benchmarks the agent on the standard benchmark. Evaluates the agent on WrappedGreenhouse

    Standard bench True:
        10 runs with 1st of March of years 2011 - 2020, averaging final balance.
    Standard bench False:
        Single run, uses config['start_date'] as the starting date.

    :param agent: The agent to bench.
    :param env_config: The config for the environment
    :param standard_bench: True to use the standard bench with 10 runs in multiple years. Defaults to True.
    :param benchmark_decade: During standard bench, evaluate for this decade. Defaults to 1.
    :param greenhouse_type: What type of greenhouse to use. Defaults to WrappedGreenhouse.
    :return: The benchmark result expressed in Final Balance in €/m²
    """
    used_config = copy.deepcopy(env_config)
    if standard_bench:
        used_config['start_date_randomization'] = "benchmark"
        used_config['start_date_decade'] = benchmark_decade
        env = greenhouse_type(used_config)
        final_balance = 0
        for i in range(10):
            _, _, episode_balance = evaluate_agent_on_env_numpy(env, agent)
            final_balance += episode_balance / 10
    else:
        used_config['start_date_randomization'] = None
        env = greenhouse_type(used_config)
        _, _, final_balance = evaluate_agent_on_env_numpy(env, agent)

    return final_balance


def benchmark_agent_with_sequence_return(
        agent: TrainerLike,
        env_config: Union[List[dict], dict],
        standard_bench: bool = True,
        benchmark_decade: int = 1,
        greenhouse_type: Type[GreenhouseNumpy] = WrappedGreenhouse
) -> (List[Sequence], float):
    """
    Benchmarks the agent on the standard benchmark. Evaluates the agent on WrappedGreenhouse

    Standard bench True:
        10 runs with 1st of March of years 2011 - 2020, averaging final balance.
    Standard bench False:
        Single run, uses config['start_date'] as the starting date.

    :param agent: The agent to bench.
    :param env_config: The config for the environment
    :param standard_bench: True to use the standard bench with 10 runs in multiple years. Defaults to True.
    :param benchmark_decade: During standard bench, evaluate for this decade. Defaults to 1.
    :param greenhouse_type: What type of greenhouse to use. Defaults to WrappedGreenhouse.
    :return: The benchmark result expressed in Final Balance in €/m²
    """
    sequences = []
    used_config = copy.deepcopy(env_config)
    if standard_bench:
        used_config['start_date_randomization'] = "benchmark"
        used_config['start_date_decade'] = benchmark_decade
        env = greenhouse_type(used_config)
        final_balance = 0
        for i in range(10):
            seq, _, episode_balance = evaluate_agent_on_env_numpy(env, agent)
            final_balance += episode_balance / 10
            sequences.append(seq)
    else:
        used_config['start_date_randomization'] = None
        env = greenhouse_type(used_config)
        seq, _, final_balance = evaluate_agent_on_env_numpy(env, agent)
        sequences.append(seq)

    return sequences, final_balance


def benchmark_agent_multiple_envs(agent: TrainerLike,
                                  env_configs: List[dict],
                                  standard_bench: bool = True,
                                  benchmark_decade: int = 1,
                                  greenhouse_type: Type[GreenhouseNumpy] = WrappedGreenhouse
                                  ) -> List[float]:
    """
    Benchmarks the agent on the standard benchmark. Evaluates the agent on WrappedGreenhouse

    Standard bench True:
        10 runs with 1st of March of years 2011 - 2020, averaging final balance.
    Standard bench False:
        Single run, uses config['start_date'] as the starting date.

    :param agent: The agent to bench.
    :param env_configs: A list of configs for each environment to evaluate
    :param standard_bench: True to use the standard bench with 10 runs in multiple years. Defaults to True.
    :param benchmark_decade: During standard bench, evaluate for this decade. Defaults to 1.
    :param greenhouse_type: What type of greenhouse to use. Defaults to WrappedGreenhouse.
    :return: The benchmark result expressed in Final Balance in €/m²
    """
    assert greenhouse_type == WrappedGreenhouse, \
        f"Other envs than WrappedGreenhouse are not supported for batching."
    return _batch_benchmark_many_configs(agent, env_configs, standard_bench)


def _batch_benchmark_many_configs(agent: TrainerLike, env_configs: List[dict], standard_bench: bool) -> List[float]:
    for i in range(len(env_configs)):
        if standard_bench:
            env_configs[i]['start_date_randomization'] = 'benchmark'
            env_configs[i]['start_date_decade'] = 1
        else:
            env_configs[i]['start_date_randomization'] = None

    envs = [WrappedGreenhouse(env_config) for env_config in env_configs]

    if standard_bench:
        return (np.sum(np.array([_run_batch_envs(agent, envs) for _ in range(10)]), axis=0) / 10).tolist()
    else:
        return _run_batch_envs(agent, envs)


def _run_batch_envs(agent: TrainerLike, envs: List[WrappedGreenhouse]):
    policy = agent.get_policy()
    if policy is None:
        policy = agent.get_policy('protagonist')

    dones = [False for _ in range(len(envs))]
    all_obs = np.stack([env.reset() for env in envs], axis=0).astype(np.float32)
    results = {}

    while sum(dones) != len(envs):
        action_batch = policy.compute_actions(all_obs, explore=False, unsquash_action=False)[0]
        if not isinstance(agent, RuleBased):
            action_batch = (action_batch + 1) / 2

        for i in range(len(envs)):
            if dones[i]:
                continue
            obs, _, done, info = envs[i].step(action_batch[i, :].reshape(6, ))
            if done:
                dones[i] = True
                results[i] = info['final_balance']
                continue
            all_obs[i, :] = obs

    return [results[i] for i in range(len(envs))]
