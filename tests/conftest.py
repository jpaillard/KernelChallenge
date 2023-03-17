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
