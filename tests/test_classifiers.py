
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from KernelChallenge.kernels import WesifeilerLehmanKernel
from KernelChallenge.KLR import KernelLogisticRegression
from KernelChallenge.preprocessing import WL_preprocess
from KernelChallenge.SVC import KernelSVC


def test_SVC(data):
    Gn, labels = data
    Gn = np.array(Gn, dtype=object)
    Gn = WL_preprocess(Gn)

    train_index, test_index = train_test_split(range(len(Gn)), test_size=0.2)
    X_train, X_test = Gn[train_index], Gn[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    WLK = WesifeilerLehmanKernel(h_iter=2, edges=True)
    clf = KernelSVC(C=1, kernel=WLK)

    clf.fit(X_train, y_train)
    clf.predict(X_test)
    assert True


def test_KLR(data):
    Gn, labels = data
    Gn = np.array(Gn, dtype=object)
    Gn = WL_preprocess(Gn)

    train_index, test_index = train_test_split(range(len(Gn)), test_size=0.2)
    X_train, X_test = Gn[train_index], Gn[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    WLK = WesifeilerLehmanKernel(h_iter=2, edges=True)
    clf = KernelLogisticRegression(reg_param=1, kernel=WLK)

    clf.fit(X_train, y_train)
    clf.predict(X_test)
    assert True
