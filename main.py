from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from KernelChallenge.classifiers import KernelSVC
from KernelChallenge.kernels import WesifeilerLehmanKernel
from KernelChallenge.preprocessing import WL_preprocess


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--h_iter', type=int, default=2)
    parser.add_argument('--c', type=float, default=1000)
    return parser.parse_args()


def fit_predict_one(Gn, labels, train_index, test_index):
    X_train, X_test = Gn[train_index], Gn[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    WLK = WesifeilerLehmanKernel(h_iter=args.h_iter)

    clf = KernelSVC(C=args.c, kernel=WLK)
    clf.fit(X_train, y_train)
    f1 = f1_score(y_test, clf.predict(X_test))
    return f1


if __name__ == '__main__':
    args = parse_args()

    data_path = Path(args.data_path)
    train_data = pd.read_pickle(data_path / 'training_data.pkl')
    labels = pd.read_pickle(data_path / 'training_labels.pkl')
    labels = 2 * labels - 1

    # Keep only the largest connected component
    # train_data = preprocess(train_data, preprocess_f0=preprocess_one)

    Gn = np.array(WL_preprocess(train_data[:args.n]))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    skf.get_n_splits(Gn, labels[:args.n])

    scores = Parallel(n_jobs=5)(delayed(
        fit_predict_one)(Gn,
                         labels,
                         train_index,
                         test_index)
        for train_index, test_index in skf.split(Gn, labels[:args.n]))

    print(
        "\nF1 score on {:d} samples using WL kernel : {:.2f}+/-{:.2f}".format(
            args.n,
            np.mean(scores),
            np.std(scores)
        ))
    print(scores)
