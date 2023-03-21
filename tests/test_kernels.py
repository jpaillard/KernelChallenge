from KernelChallenge.kernels import WesifeilerLehmanKernel
from KernelChallenge.kernels import gramMatrix
from KernelChallenge.preprocessing import WL_preprocess


def test_WL_subtree(WL_graphs):
    """
    Test inspired by the example in the paper Figure 2
    """
    Gn = WL_preprocess(WL_graphs)
    WLK = WesifeilerLehmanKernel(h_iter=1)
    feat_vectors = WLK.fit_subtree(Gn)

    fest_test = WLK.predict(Gn)
    assert feat_vectors.shape == fest_test.shape

    K = gramMatrix(feat_vectors, feat_vectors)
    assert K[0, 1] == 11
