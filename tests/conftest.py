import networkx as nx
import numpy as np
import pytest


def make_one_graph(n=10):
    G = nx.Graph()
    G.add_nodes_from([(i, {"labels": [np.random.randint(50)]})
                     for i in range(n)])
    G.add_edges_from([(i, i + 1) for i in range(n - 1)])
    return G


@pytest.fixture(scope='module')
def dummy_graphs():
    return [make_one_graph() for _ in range(10)]


@pytest.fixture(scope='module')
def WL_graphs():
    G1 = nx.Graph()
    G2 = nx.Graph()

    G1.add_nodes_from([
        (1, {"labels": [1]}),
        (2, {"labels": [1]}),
        (3, {"labels": [4]}),
        (4, {"labels": [5]}),
        (5, {"labels": [2]}),
        (6, {"labels": [3]})
    ])
    G2.add_nodes_from([
        (1, {"labels": [2]}),
        (2, {"labels": [1]}),
        (3, {"labels": [4]}),
        (4, {"labels": [2]}),
        (5, {"labels": [5]}),
        (6, {"labels": [3]})
    ])
    G1.add_edges_from([
        (1, 3, {"labels": [1]}),
        (2, 3, {"labels": [1]}),
        (3, 4, {"labels": [1]}),
        (3, 6, {"labels": [1]}),
        (4, 5, {"labels": [1]}),
        (4, 6, {"labels": [1]}),
        (5, 6, {"labels": [1]}),
    ])
    G2.add_edges_from([
        (1, 6, {"labels": [1]}),
        (2, 3, {"labels": [1]}),
        (3, 4, {"labels": [1]}),
        (3, 5, {"labels": [1]}),
        (3, 6, {"labels": [1]}),
        (4, 5, {"labels": [1]}),
        (5, 6, {"labels": [1]}),
    ])
    return [G1, G2]
