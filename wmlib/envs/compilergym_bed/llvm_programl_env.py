import numpy as np
from compiler_gym.wrappers import CompilerEnvWrapper
from compiler_gym.util.gym_type_hints import ActionType
from typing import *

import networkx as nx
try:
    import torch_geometric as pyg
except Exception:
    pyg = None

from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.reportviews import EdgeView, NodeView

import json
import os
from pathlib import Path

_dir = os.path.join(Path(__file__).parent, 'vocab.json')
with open(_dir, 'r') as fp:
    VOCAB = json.load(fp)

NODE_FEATURES = ['text', 'type']
MAX_TEXT, MAX_TYPE = len(VOCAB), 3

EDGE_FEATURES = ['flow', 'position']
MAX_FLOW, MAX_POS = 3, 5120

# TODO: graph too large, subsampling
MAX_NODES = int(1e4)
MAX_EDGES = int(1e4)


def parse_nodes(ns: NodeView, return_attr=False):
    # in-place
    x = []
    for nid in ns:
        n = ns[nid]
        n.pop('function', None)
        n.pop('block', None)
        n.pop('features', None)
        n['text'] = VOCAB.get(n['text'], MAX_TEXT)
        if return_attr:
            x.append(np.array([n['text'], n['type']]))
    return x


def parse_edges(es: EdgeView, return_attr=False):
    # in-place
    x = []
    for eid in es:
        e = es[eid]
        e['position'] = min(e['position'], MAX_POS)

        if return_attr:
            x.append(np.array([e['flow'], e['position']]))

    return x


def parse_graph(g: MultiDiGraph):
    # TODO: want to avoid for loop
    x = parse_nodes(g.nodes, return_attr=True)
    edge_attr = parse_edges(g.edges, return_attr=True)

    g = nx.DiGraph(g)
    edge_index = list(g.edges)

    return np.array(x), np.transpose(np.array(edge_index)), np.array(edge_attr)


class GraphSizeExceededException(Exception):
    pass


class ProgramlWrapper(CompilerEnvWrapper):
    def __init__(self, env, max_node=MAX_NODES, max_edge=MAX_EDGES, ignore_size_limit=False):
        super().__init__(env)
        self.max_node, self.max_edge = max_node, max_edge
        self.ignore_size_limit = ignore_size_limit

    def set_ignore_programl_size_limit(self, flag):
        self.ignore_size_limit = flag

    def multistep(
        self,
        actions: List[ActionType],
        **kwargs,
    ):
        obs, rew, done, info = super().multistep(actions, **kwargs)
        return parse_graph(obs), rew, done, info

    def reset(self, *args, **kwargs):
        obs: MultiDiGraph = self.env.reset(*args, **kwargs)
        if not self.ignore_size_limit and \
            (obs.number_of_nodes() > self.max_node or
             obs.number_of_edges() > self.max_edge):
            print("graph size", obs.number_of_nodes(), obs.number_of_edges(), "exceeded size limit.")
            raise GraphSizeExceededException()
        else:
            # print(obs.number_of_nodes(), obs.number_of_edges())
            obs = parse_graph(obs)
            return obs
