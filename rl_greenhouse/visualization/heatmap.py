import copy
import math
from os.path import join
from pathlib import Path
from typing import Dict, Tuple, List, Type, Union

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ray.rllib.agents import Trainer

from rl_greenhouse.greenhouse import DEFAULT_GREENHOUSE_CONFIG, GreenhouseNumpy, WrappedGreenhouse
from rl_greenhouse.utils.evaluation import benchmark_agent_multiple_envs


def create_heatmap(
        agent: Trainer,
        env_config: dict,
        variables_to_range: Dict[str, Tuple[float, float]],
        nr_steps: int = 11,
        bench: bool = True,
        image_name: str = None,
        greenhouse_type: Type[GreenhouseNumpy] = WrappedGreenhouse
) -> None:
    """
    Creates a heatmap of the agent its performance vs varying settings for hyperparameters.

    :param agent: The agent to use for evaluation.
    :param env_config: Config overrides for the greenhouse. Variables will be injected into this dict.
    :param variables_to_range: Maps a config variable to a range. Evaluates between these ranges relative to the center.
    :param nr_steps: Number of steps along a single axis. Defaults to 4.
    :param bench:
    :param image_name: Base name of the file where the image is stored. Set to None to not save anything.
        Defaults to None.
    :param greenhouse_type: What type of greenhouse to use. Defaults to WrappedGreenhouse.
    """
    first_var, second_var = variables_to_range.keys()
    env_configs, first_values, second_values = \
        _create_env_config_batch_for_heatmap(env_config, variables_to_range, first_var, second_var, nr_steps)

    result_list = benchmark_agent_multiple_envs(agent, env_configs, bench)
    results = _transform_list_of_results_into_result_dict(result_list)

    print("(", sep='\n\t')
    print("\t" + str(results), first_values.tolist(), second_values.tolist(), f"'{first_var}'", f"'{second_var}'",
          sep=',\n\t')
    print(")")

    heatmap(results, first_values, second_values, first_var, second_var)
    if image_name:
        plt.savefig(create_name_heatmap(image_name, str(agent), [first_var, second_var], nr_steps, bench))
    plt.show()


def _create_env_config_batch_for_heatmap(
        env_config: dict,
        variables_to_range: Dict[str, Tuple[float, float]],
        first_var: str,
        second_var: str,
        nr_steps: int,
) -> (List[Dict[str, float]], np.ndarray, np.ndarray):
    """
    First entry has the default value, others have the other variations
    """
    first_min, first_max = variables_to_range[first_var]
    second_min, second_max = variables_to_range[second_var]
    first_values = np.linspace(first_max, first_min, num=nr_steps)
    second_values = np.linspace(second_min, second_max, num=nr_steps)

    env_configs = []
    for first_val in first_values:
        for second_val in second_values:
            used_config = copy.deepcopy(env_config)
            used_config[first_var] = first_val
            used_config[second_var] = second_val
            env_configs.append(used_config)
    return env_configs, first_values, second_values


def _transform_list_of_results_into_result_dict(results: List[float]) -> List[List[float]]:
    square_side_length = int(math.sqrt(len(results)))
    nested_list = []
    for i in range(square_side_length):
        nested_list.append([])
        for j in range(square_side_length):
            nested_list[-1].append(results[i * square_side_length + j])
    return nested_list


def heatmap(
        results: List[List[float]],
        first_values: Union[List[float], np.ndarray],
        second_values: Union[List[float], np.ndarray],
        first_var: str,
        second_var: str,
        title: str = None,
        color_min_max: Tuple[float, float] = None,
        existing_axes = None,
        add_cbar: bool = True,
        cbar_horizontal: bool = False,
        add_legend: bool = True,
) -> None:
    """
    Produces a heatmap of the results.

    :param results: List of lists
    :param first_values: y-values on the y-axis
    :param second_values: x-values on the x-axis
    :param first_var: y variable name
    :param second_var: x variable name
    """
    y_ticks = [round(y, 3) for y in first_values]
    x_ticks = [round(x, 3) for x in second_values]
    y_ctr = len(first_values) / 2
    x_ctr = len(second_values) / 2

    if color_min_max is not None:
        heatmap_kwargs = {'vmin': color_min_max[0], 'vmax': color_min_max[1]}
    else:
        heatmap_kwargs = {}

    if cbar_horizontal and add_cbar:
        heatmap_kwargs['cbar_kws'] = {'orientation': "horizontal"}
    if not add_cbar:
        heatmap_kwargs["cbar"] = False

    if existing_axes is None:
        plt.cla()
        plt.clf()
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(aspect='equal')
    else:
        ax = existing_axes

    hax = sns.heatmap(results, xticklabels=x_ticks, yticklabels=y_ticks, square=True, ax=ax,
                      cmap=sns.color_palette("coolwarm", as_cmap=True,), **heatmap_kwargs)

    if add_cbar:
        hax.collections[0].colorbar.set_label("Benchmark Final Balance [€/m²]")
    ax.scatter(x=y_ctr, y=x_ctr, marker="x", color="black", s=80, label="Original GH")
    ax.axhline(y_ctr, linestyle='--', color='black', alpha=0.2)
    ax.axvline(x_ctr, linestyle='--', color='black', alpha=0.2)

    ax.set_ylabel(first_var)
    ax.set_xlabel(second_var)
    if title:
        ax.set_title(title, loc='right', fontsize=9)

    if add_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(-0.30, 1.16), fontsize=9)
    plt.tight_layout()
    return hax


def create_name_heatmap(agent_name: str, algorithm_name: str, param_names: List[str], nr_steps: int,
                        bench: bool) -> str:
    if bench:
        benched = "benched"
    else:
        benched = "non-benched"
    name = \
        f"heatmap-{benched}-{agent_name}-{algorithm_name}-{param_names[0]}-{param_names[1]}-{nr_steps}x{nr_steps}.png"
    print(name)
    return join(Path("~/ray_results").expanduser(), name)


def create_ranges_25_25(variables: Tuple[str, str]) -> (Dict[str, Tuple[float, float]]):
    """

    :param variables:
    :return:
    """
    ranges = {}
    for var in variables:
        val = DEFAULT_GREENHOUSE_CONFIG[var]
        ranges[var] = []
        for multiplier in (0.75, 1.25):
            ranges[var].append(multiplier * val)
        ranges[var] = tuple(ranges[var])
    return ranges


def create_ranges_50_50(variables: Tuple[str, str]) -> (Dict[str, Tuple[float, float]]):
    """

    :param variables:
    :return:
    """
    ranges = {}
    for var in variables:
        val = DEFAULT_GREENHOUSE_CONFIG[var]
        ranges[var] = []
        for multiplier in (0.5, 1.5):
            ranges[var].append(multiplier * val)
        ranges[var] = tuple(ranges[var])
    return ranges


def create_ranges_100_100(variables: Tuple[str, str]) -> (Dict[str, Tuple[float, float]]):
    """

    :param variables:
    :return:
    """
    ranges = {}
    for var in variables:
        val = DEFAULT_GREENHOUSE_CONFIG[var]
        ranges[var] = []
        for multiplier in (0, 2):
            ranges[var].append(multiplier * val)
        ranges[var] = tuple(ranges[var])
    return ranges
