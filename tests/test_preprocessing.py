import networkx as nx

from KernelChallenge.preprocessing import preprocess
from KernelChallenge.preprocessing import preprocess_one


def test_preprocess(dummy_molecule):
    preprocessed = preprocess([dummy_molecule], preprocess_f0=preprocess_one)
    assert isinstance(preprocessed[0], nx.Graph)
