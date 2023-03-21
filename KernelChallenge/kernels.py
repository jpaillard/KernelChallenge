from copy import deepcopy

import networkx as nx
import numpy as np
from tqdm import tqdm


def gramMatrix(X1, X2: np.ndarray, ):
    """
    Compute the Gram matrix of a feature matrix.
    Implemented outised to allow for different kernels.
    """
    return np.dot(X1, X2.T)


class WesifeilerLehmanKernel():

    def __init__(self, h_iter: int = 2):
        self.h_iter = h_iter
        self.unique_labels_h = dict()
        self.unique_labels_count = 0

    def fit_subtree(self, G: list):
        """
        Weisfeiler-Lehman subtree kernel as detailed in Algorithm 2:
        https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

        """
        Gn = deepcopy(G)
        # 0: Initialization
        unique_labels_0 = []
        for g in Gn:
            unique_labels_0 += list(nx.get_node_attributes(
                g, name='labels').values())

        unique_labels_0 = np.unique(unique_labels_0)
        unique_labels = {v: i for i, v in enumerate(unique_labels_0)}

        feat_vector_h = np.zeros((len(Gn), len(unique_labels_0)), dtype=int)
        for i, g in enumerate(Gn):
            for node, label in g.nodes(data='labels'):
                feat_vector_h[i, int(unique_labels[label])] += 1
            feat_vector = feat_vector_h
        self.unique_labels_h[0] = unique_labels

        # Can be parallelized over depth (h_iter) and/or graphs (Gn)
        for h in range(self.h_iter):
            unique_labels = dict()
            unique_labels_count = 0

            # iterate over all graphs
            for idx, g in enumerate(tqdm(Gn)):
                gc = g.copy()
                for node, label in g.nodes(data='labels'):

                    # 1: Multiset-label determination
                    multiset_tmp = sorted(
                        [int(g.nodes[neighbor]['labels'])
                         for neighbor in g[node]]
                    )

                    # 2: Sorting each multiset
                    s_i = label + ''.join(str(x) for x in multiset_tmp)

                    # 3: Label compression
                    if s_i in unique_labels.keys():
                        cp_label = unique_labels[s_i]
                    else:
                        unique_labels.update({s_i: unique_labels_count})
                        cp_label = unique_labels_count
                        unique_labels_count += 1

                    # 4: Relabeling tmp
                    gc.nodes[node]['labels'] = str(cp_label)
                # 4: Relabeling
                Gn[idx] = gc

            # Compute feature vector at step h
            feat_vector_h = np.zeros((len(Gn), unique_labels_count), dtype=int)
            for i, g in enumerate(Gn):
                for node, label in g.nodes(data='labels'):
                    feat_vector_h[i, int(label)] += 1

            self.unique_labels_h[h + 1] = unique_labels

            # Stack feature vectors
            feat_vector = np.hstack((feat_vector, feat_vector_h))
        return feat_vector

    def predict(self, G):
        """
        Same as above but ensures that the feature vectors have the same shape
        in train and test.
        """
        Gn = deepcopy(G)

        feat_vector_h = np.zeros(
            (len(Gn), len(self.unique_labels_h[0])), dtype=int)

        for i, g in enumerate(Gn):
            for node, label in g.nodes(data='labels'):
                if label in self.unique_labels_h[0].keys():
                    feat_vector_h[i, int(self.unique_labels_h[0][label])] += 1
            feat_vector = feat_vector_h

        # Can be parallelized over depth (h_iter) and/or graphs (Gn)
        for h in range(self.h_iter):
            unique_labels = dict()
            unique_labels_count = 0
            s_i_list = []

            # iterate over all graphs
            for idx, g in enumerate(tqdm(Gn)):
                gc = g.copy()
                s_i_tmp = []
                for node, label in g.nodes(data='labels'):

                    # 1: Multiset-label determination
                    multiset_tmp = sorted(
                        [int(g.nodes[neighbor]['labels'])
                         for neighbor in g[node]]
                    )

                    # 2: Sorting each multiset
                    s_i = label + ''.join(str(x) for x in multiset_tmp)
                    s_i_tmp.append(s_i)

                    # 3: Label compression
                    if s_i in unique_labels.keys():
                        cp_label = unique_labels[s_i]
                    else:
                        unique_labels.update({s_i: unique_labels_count})
                        cp_label = unique_labels_count
                        unique_labels_count += 1

                    # 4: Relabeling tmp
                    gc.nodes[node]['labels'] = str(cp_label)
                s_i_list.append(s_i_tmp)
                # 4: Relabeling
                Gn[idx] = gc

            # Compute feature vector at step h
            feat_vector_h = np.zeros(
                (len(Gn), len(self.unique_labels_h[h + 1])), dtype=int)
            for i, s_i_tmp in enumerate(s_i_list):
                for s_i in s_i_tmp:
                    if s_i in self.unique_labels_h[h + 1].keys():
                        feat_vector_h[i, int(
                            self.unique_labels_h[h + 1][s_i])] += 1

            # Stack feature vectors
            feat_vector = np.hstack((feat_vector, feat_vector_h))
        return feat_vector
