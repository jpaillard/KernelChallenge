from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from KernelChallenge.kernels import WesifeilerLehmanKernel
from KernelChallenge.kernels import gramMatrix
from KernelChallenge.preprocessing import WL_preprocess


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--h_iter', type=int, default=2)
    parser.add_argument('--c', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_path = Path(args.data_path)
    train_data = pd.read_pickle(data_path / 'training_data.pkl')
    labels = pd.read_pickle(data_path / 'training_labels.pkl')

    Gn = WL_preprocess(train_data[:args.n])
    print(len(Gn))
    X_train, X_test, y_train, y_test = train_test_split(
        Gn,
        labels[:args.n],
        test_size=0.2,
        random_state=0,
        stratify=labels[:args.n]
    )

    WLK = WesifeilerLehmanKernel(h_iter=args.h_iter)
    feat_train = WLK.fit_subtree(X_train)
    feat_test = WLK.predict(X_test)

    K_train = gramMatrix(feat_train, feat_train)
    K_test = gramMatrix(feat_test, feat_train)
    clf = SVC(C=args.c, kernel='precomputed')
    clf.fit(K_train, y_train)
    print("\nF1 score on {:d} samples using WL kernel :  {:.2f}".format(
        args.n, f1_score(y_test, clf.predict(K_test))))
