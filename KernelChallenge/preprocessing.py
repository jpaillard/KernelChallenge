
import networkx as nx
from joblib import Parallel
from joblib import delayed


def preprocess_one(G: nx.Graph):
    '''
    Extracts the largest connected component of a graph
    '''
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(Gcc[0])


def preprocess(lisg_G: list,
               preprocess_f0,
               n_jobs: int = 1,
               **kwargs):
    '''
    Preprocess all graphs in parallel
    '''
    output = Parallel(n_jobs=n_jobs)(delayed(preprocess_f0)(G, **kwargs)
                                     for G in lisg_G)

    return output
