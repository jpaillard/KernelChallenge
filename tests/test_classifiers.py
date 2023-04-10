import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from KernelChallenge.kernels import WesifeilerLehmanKernel
from KernelChallenge.SVC import KernelSVC


def test_SVC(data):
    Gn, labels = data
    print(Gn)
    Gn = np.array(Gn, dtype=object)

    train_index, test_index = train_test_split(range(len(Gn)), test_size=0.2)
    X_train, X_test = Gn[train_index], Gn[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    WLK = WesifeilerLehmanKernel(h_iter=2, edges=True)
    clf = KernelSVC(C=1, kernel=WLK)

    predict_train = clf.fit(X_train, y_train)
    score_train = roc_auc_score(y_train, predict_train)
    score_test = roc_auc_score(y_test, clf.predict(X_test))
    assert True


# %%


# %%
data = pd.read_pickle('tests/data.pkl')
labels = pd.read_pickle('tests/labels.pkl')
