import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
# from joblib import Parallel
# from joblib import delayed
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from KernelChallenge.classifiers import KernelSVC
from KernelChallenge.kernels import WesifeilerLehmanKernel
from KernelChallenge.preprocessing import WL_preprocess

# from sklearn.svm import SVC


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--h_iter', type=int, default=2)
    parser.add_argument('--c', type=int, default=1000)
    parser.add_argument('--edges', action='store_true')
    parser.add_argument('--submit', action='store_true')
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
    test_data = pd.read_pickle(data_path / 'training_data.pkl')
    labels = pd.read_pickle(data_path / 'training_labels.pkl')
    labels = 2 * labels - 1

    Gn = WL_preprocess(random.sample(train_data, args.n))
    # Gn = WL_preprocess(train_data[:args.n])
    Gn = np.array(Gn)

    print(len(Gn))
    X_train, X_test, y_train, y_test = train_test_split(
        Gn,
        labels[:args.n],
        test_size=0.2,
        random_state=0,
        stratify=labels[:args.n], shuffle=True
    )

    WLK = WesifeilerLehmanKernel(h_iter=args.h_iter, edges=args.edges)

    clf = KernelSVC(C=args.c, kernel=WLK)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=========================================")
    print("\nF1 score on {:d} samples using WL kernel :  {:.2f}".format(
        args.n, f1_score(y_test, y_pred)))
    print("\nAccuracy score on {:d} samples using WL kernel :  {:.2f}".format(
        args.n, accuracy_score(y_test, y_pred)))
    print("\nRecall score on {:d} samples using WL kernel :  {:.2f}".format(
        args.n, recall_score(y_test, y_pred)))
    print("\nPrecision score on {:d} samples using WL kernel :  {:.2f}".format(
        args.n, precision_score(y_test, y_pred)))
    print("\n=========================================")

    # TODO  : Retrain on the whole training set using CValidated parameters

    if args.submit:
        Gn = WL_preprocess(test_data)
        y_pred = clf.predict(Gn)
        pd.DataFrame(y_pred).to_csv('y_pred.csv', index=False)
