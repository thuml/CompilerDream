import numpy as np
import torch

import os
import random
import math
from typing import Tuple
from typing import List
try:
    from torch_geometric.data import Data, Batch
except Exception:
    pass
from collections import defaultdict
from functools import reduce


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def snapshot_src(src, target, exclude_from):
    make_dir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")


def fix_range(start: int, total: int, max_len: int, fix_overflow: bool = True) -> Tuple[int, int]:
    if max_len <= 0:
        max_len = 2**32 - 1
    if total < 0:
        total = (max_len - start) if start < max_len else max_len
        # print(f"[Info] Use all data of dataset (total {max_len}).")
    if start + total > max_len and fix_overflow:
        new_start = max(0, max_len - total)
        new_total = min(max_len - new_start, total)
        print(
            f"[Warning] benchmarksBuilder: selected range [{start},{start+total})" +
            f" exceeded dataset range [0,{max_len}), fix to [{new_start}, {new_start+new_total}).")
        start, total = new_start, new_total
    return start, total


def geom_mean(input_list: List):
    output_list = np.array(input_list, dtype=float)
    if not len(output_list) or np.min(output_list) <0.0:
        return 0.0
    return (output_list ** (1 / len(output_list))).prod()


def iqm(input_list: List):
    l = len(input_list) // 4
    return np.mean(sorted(input_list)[l:-l])


def triplet_compare(oz, final):
    if oz < final:
        return -1
    elif oz > final:
        return 1
    else:
        return 0


def symlog(y):
    if isinstance(y, np.ndarray):
        return np.sign(y) * np.log(1 + np.abs(y))
    elif isinstance(y, torch.Tensor):
        return torch.sgn(y) * torch.log(1 + torch.abs(y))
    elif isinstance(y, float):
        return math.copysign(1, y) * math.log(1 + abs(y))
    else:
        raise NotImplementedError


def symexp(y):
    if isinstance(y, np.ndarray):
        return np.sign(y) * (np.exp(np.abs(y)) - 1)
    elif isinstance(y, torch.Tensor):
        return torch.sgn(y) * (torch.exp(torch.abs(y)) - 1)
    elif isinstance(y, float):
        return math.copysign(1, y) * (math.exp(abs(y)) - 1)
    else:
        raise NotImplementedError

def obs_to_device(obs, device):
    def convert_single_dict(obs:dict):
        obs_device = {}
        for key, value in obs.items():
            if 'programl' == key:
                obs_device['programl'] = [
                    torch.from_numpy(np.array(value[0])).long().to(device),
                    torch.from_numpy(np.array(value[1])).long().to(device),
                    torch.from_numpy(np.array(value[2])).long().to(device)
                ]
            else:
                obs_device[key] = torch.from_numpy(np.array(value)).float().to(device)
        return obs_device
    if isinstance(obs, dict): # sigle data
        return convert_single_dict(obs)
    elif isinstance(obs, list) or isinstance(obs, np.ndarray): # batched data
        return [convert_single_dict(d) for d in obs]

def parse_obs(batch_obs: list, device):
    with torch.no_grad():
        vector_obs = []
        graph_obs = {}

        # assume all obs has the same keys
        for k in batch_obs[0]:
            if k == 'programl':
                data = []
                for obs in batch_obs:
                    x = torch.from_numpy(np.array(obs[k][0])).long().to(device)
                    edge_index = torch.from_numpy(np.array(obs[k][1])).long().to(device)
                    edge_attr = torch.from_numpy(np.array(obs[k][2])).long().to(device)
                    data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
                graph_obs = Batch.from_data_list(data)
            else:
                v = np.array([obs[k] for obs in batch_obs])
                v = torch.from_numpy(v).float().to(device)
                vector_obs.append(v)

        vector_obs = torch.cat(vector_obs, dim=-1)
        return vector_obs, graph_obs if 'programl' in batch_obs[0] else None

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def find_file(path, filename):
    while path and path !='/':
        if os.path.exists(os.path.join(path, filename)):
            return os.path.join(path, filename)
        path = os.path.split(path)[0]
    return None

import sys
import pdb


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
    
        
class random_with_replacement(object):
    def __init__(self, data:list, seed) -> None:
        self.data = data
        self._random_state = np.random.RandomState(seed)
        self.seed = seed

    def __iter__(self):
        return self
    
    def __next__(self):
        return self._random_state.choice(self.data, replace=True)
    
class random_id_iter(object):
    def __init__(self, pattern_str:str, st, ed, step, seed) -> None:
        self.pattern = pattern_str
        self._random_state = np.random.RandomState(seed)
        self.seed = seed
        self.id_geneator = lambda: st + self._random_state.randint(0, (ed-st+step-1)//step)*step
    def __iter__(self):
        return self
    def __next__(self):
        return f"{self.pattern}/{self.id_geneator()}"

# Debugs
def assert_successive_obs(obs, next_obs, action):
    assert type(obs) is type(next_obs), f"type of obs({type(obs)}) not match with next_obs({type(next_obs)})"
    _delta = np.zeros(42)
    _delta[action] = 1.0
    if isinstance(obs, dict):
        assert (obs["action_histogram"] - next_obs['action_histogram'] == _delta / 45).any()
    elif isinstance(obs, list) or isinstance(obs, np.ndarray):    
        assert((np.array(obs[-42:]) - np.array(next_obs[-42:]) == _delta / 45).any())
    else:
        assert False, f"observation type {type(obs)} not supported."

def POJUrisReordered(uris):
    '''
    Reorder the POJ dataset uris so that first 50 benchmarks (test set)
    are from different problem and 50-100th benchmarks(valid set) are also from different
    prolems. 
    '''
    d = defaultdict(list)
    for uri in uris:
        d[int(uri.split('/')[-2])].append(uri)
    l = [d[i][0] for i in range(1, 51)] + [d[i][0] for i in range(51, 101)] \
        + reduce(lambda x,y:x+y, [d[i][1:] for i in range(1, 101)])\
        + reduce(lambda x,y:x+y, [d[i] for i in range(101, 105)])
    return  l 