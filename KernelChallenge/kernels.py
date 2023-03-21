import numpy as np


def weisfeilerLehman_subtreeKernel(Gn: list, h_iter: int = 2):
    """
    Weisfeiler-Lehman subtree kernel as detailed in Algorithm 2:
    https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

    """
    # 0: Initialization
    unique_labels_0 = np.unique(
        [list(dict(g.nodes(data='labels')).values()) for g in Gn])
    unique_labels = {v: i for i, v in enumerate(unique_labels_0)}

    feat_vector_h = np.zeros((len(Gn), len(unique_labels_0)), dtype=int)
    for i, g in enumerate(Gn):
        for node, label in g.nodes(data='labels'):
            feat_vector_h[i, int(unique_labels[label])] += 1
        feat_vector = feat_vector_h

    # Can be parallelized over depth (h_iter) and/or graphs (Gn)
    for h in range(h_iter):
        unique_labels = dict()
        unique_labels_count = 0

        # iterate over all graphs
        for idx, g in enumerate(Gn):
            gc = g.copy()
            for node, label in g.nodes(data='labels'):

                # 1: Multiset-label determination
                multiset_tmp = sorted(
                    [int(g.nodes[neighbor]['labels']) for neighbor in g[node]])

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

        # Stack feature vectors
        feat_vector = np.hstack((feat_vector, feat_vector_h))
    return feat_vector


def gramMatrix(features: np.ndarray):
    """
    Compute the Gram matrix of a feature matrix.
    Implemented outised to allow for different kernels.
    """
    return np.dot(features, features.T)
