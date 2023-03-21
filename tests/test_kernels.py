import numpy as np

from KernelChallenge.kernels import weisfeilerLehman_subtreeKernel
from KernelChallenge.preprocessing import WL_preprocess


def test_WL_subtree(WL_graphs):
    """
    Test inspired by the example in the paper Figure 2
    """
    feat_vectors = weisfeilerLehman_subtreeKernel(
        WL_preprocess(WL_graphs),
        h_iter=1
    )
    assert np.inner(feat_vectors[0], feat_vectors[1]) == 11
