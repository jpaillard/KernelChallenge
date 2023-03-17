import networkx as nx
import pytest


@pytest.fixture(scope='module')
def dummy_molecule():
    G = nx.Graph()
    G.add_nodes_from([
        (1, {"labels": [1]}),
        (2, {"labels": [0]}),
        (3, {"labels": [1]}),
        (4, {"labels": [1]}),
        (5, {"labels": [5]}),
    ])
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (4, 5)])
    return G
