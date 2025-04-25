import copy
import random
from typing import List, Callable, Union

from rl_greenhouse.greenhouse.types import Plant, Sequence
from rl_greenhouse.greenhouse import Greenhouse
from rl_greenhouse.utils.evaluation import play_actions_on_env
from rl_greenhouse.visualization.time_series import compare_sequences

SEQUENCE_ERROR_METRIC = Callable[[Sequence, Sequence, List[str], bool], float]


def show_n_replay_samples(sequences: List[Sequence], config: dict, variables: List[str] = None, nr_samples: int = None):
    """Show n sample"""
    random.seed(0)
    for sequence in (sequences if not nr_samples else random.choices(sequences, k=nr_samples)):
        show_replay_sample(sequence, config, variables=variables)


def show_replay_sample(sequences: Union[List[Sequence], Sequence], config: dict, variables: List[str] = None, idx=0, **kwargs):
    if isinstance(sequences, list):
        sequence_sample = sequences[idx]
    else:
        sequence_sample = sequences
    sample_config = copy.deepcopy(config)
    sample_config = inject_start_and_replay_settings_into_config(sample_config, sequence_sample)

    greenhouse = Greenhouse(sample_config)
    own_sequence, _, _ = play_actions_on_env(greenhouse, sequence_sample.actions, stop_when_done=False)
    compare_sequences([sequence_sample, own_sequence], variables=variables, labels=["INTKAM / KASPRO", "Our Model"], **kwargs)
    return sequence_sample, own_sequence


def inject_start_and_replay_settings_into_config(config: dict, sequence: Sequence) -> dict:
    config["plants_to_replay"] = sequence.plants
    config["states_to_replay"] = sequence.states
    config["weather_to_replay"] = sequence.weathers
    # config['starting_info'] = sequence.infos[0]
    config['starting_weather'] = sequence.weathers[0]
    config["starting_state"] = sequence.states[0]

    # Plants tend to have a mass of 0 from the data. If this is true then initialize the default plant.
    first_plant = sequence.plants[0]
    if first_plant.mass > 2:
        config["starting_plant"] = first_plant
    else:
        config["starting_plant"] = Plant()
    return config
