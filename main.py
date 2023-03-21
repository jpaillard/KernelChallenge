from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from KernelChallenge.kernels import WesifeilerLehmanKernel
from KernelChallenge.kernels import gramMatrix
from KernelChallenge.preprocessing import WL_preprocess

data_path = Path('data/')
train_data = pd.read_pickle(data_path / 'training_data.pkl')
labels = pd.read_pickle(data_path / 'training_labels.pkl')
N = 1000

if __name__ == '__main__':
    WLK = WesifeilerLehmanKernel(h_iter=1)

    Gn = WL_preprocess(train_data[:N])
    X_train, X_test, y_train, y_test = train_test_split(
        Gn, labels[:N], test_size=0.2, random_state=0, stratify=labels[:N])

    WLK = WesifeilerLehmanKernel(h_iter=2)
    feat_train = WLK.fit_subtree(X_train)
    feat_test = WLK.predict(X_test)

    K_train = gramMatrix(feat_train, feat_train)
    K_test = gramMatrix(feat_test, feat_train)
    clf = SVC(C=1000, kernel='precomputed')
    clf.fit(K_train, y_train)
    print("F1 score on {:d} samples using WL kernel :  {:.2f}".format(
        N, f1_score(y_test, clf.predict(K_test))))
