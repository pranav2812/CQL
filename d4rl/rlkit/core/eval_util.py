"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np

import rlkit.pythonplusplus as ppp

# For normalized score in eval list
import d4rl

def get_normalized_score(env_name, score):
    ref_min_score = d4rl.infos.REF_MIN_SCORE[env_name]
    ref_max_score = d4rl.infos.REF_MAX_SCORE[env_name]
    return (score - ref_min_score) / (ref_max_score - ref_min_score)


def save_txts(avg_reward, d4rl_score, episode_return_set, episode_d4rl_set, envName, envSeed):
    file_name = f"CQL_{envName}_{envSeed}"
    def save_file(_type, arr):
        file = open(f"./results_new/CQL/{file_name}_{_type}.txt", "a+")
        np.savetxt(file, arr)
        file.close()
    save_file('d4rl', d4rl_score)
    save_file('return', avg_reward)
    save_file('d4rl_epi', episode_d4rl_set)
    save_file('return_epi', episode_return_set)

def get_generic_path_information(paths, stat_prefix='', isThisEval=False, envName=None, envSeed=0):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """

    # I have 10 paths for evaulation now, variable for exploration
    # I need d4rl_score, avg_reward, episode_d4rl_set, episode_return_set

    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]
    
    if isThisEval:
        # make modifications for resulting values
        avg_reward = sum(returns)/10
        d4rl_score = get_normalized_score(envName, avg_reward)
        episode_return_set = np.array(returns)
        episode_d4rl_set = np.array([get_normalized_score(envName, ii) for ii in returns])
        
        save_txts(avg_reward, d4rl_score, episode_return_set, episode_d4rl_set, envName=envName, envSeed=envSeed)
        # # Logger of Dict cannot handle arrays
        # statistics.update({'d4rl_score':d4rl_score})
        # statistics.update({'avg_reward':avg_reward})
        # statistics.update({'episode_d4rl_set':episode_d4rl_set})
        # statistics.update({'episode_return_set':episode_return_set})


    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)
    statistics[stat_prefix + 'Average Returns'] = get_average_returns(paths)

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats
