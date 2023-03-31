# import cProfile
from argparse import ArgumentParser
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
# from joblib import Parallel
# from joblib import delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from KernelChallenge.classifiers import KernelSVC
from KernelChallenge.KLR import KernelLogisticRegression
from KernelChallenge.kernels import WesifeilerLehmanKernel
# from KernelChallenge.kernels import gramMatrix
from KernelChallenge.preprocessing import WL_preprocess

from wwl import wwl


# from sklearn.svm import SVC


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--h_iter', type=int, default=2)
    parser.add_argument('--c', type=int, default=1000)
    parser.add_argument('--method', type=str, default="SVC")
    parser.add_argument('--edges', action='store_true')
    parser.add_argument('--submit', action='store_true')
    return parser.parse_args()


def fit_predict_one(Gn, labels, train_index, test_index,
                    metric=roc_auc_score, c=100, h_iter=2, edges=False, method = "SVC"):
    X_train, X_test = Gn[train_index], Gn[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    
    
    
    
    WLK = WesifeilerLehmanKernel(h_iter=h_iter, edges=edges)

    if method == "SVC":
        clf = KernelSVC(C=c, kernel=WLK)

    elif method == "KLR":
        clf = KernelLogisticRegression(reg_param= c, kernel = WLK)
    
    clf.fit(X_train, y_train)
    score = metric(y_test, clf.predict(X_test))
    print("AUC score on test set : {:.2f}".format(score))
    return score


if __name__ == '__main__':
    args = parse_args()

    data_path = Path(args.data_path)
    train_data = pd.read_pickle(data_path / 'training_data.pkl')
    test_data = pd.read_pickle(data_path / 'training_data.pkl')
    labels = pd.read_pickle(data_path / 'training_labels.pkl')
    labels = 2 * labels - 1

    np.random.seed(0)
    idx = np.random.choice(len(train_data), args.n, replace=False)
    Gn = np.array(train_data)[idx]
    labels = labels[idx]

    Gn = WL_preprocess(Gn)

    #  klf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) Think about performing K fold

    N = len(Gn)
    t0 = time()
    scores = fit_predict_one(Gn,
                             labels,
                             np.arange(N * 4 // 5),
                             np.arange(N * 4 // 5, N),
                             c=args.c,
                             h_iter=args.h_iter,
                             edges=args.edges,
                             method = args.method)

    print("=========================================")
    print(f"C = {args.c}")
    print(f"\nAUC score on {args.n} samples using WL kernel and {args.method} :  {np.mean(scores)} ")
    print("Time : {:.2f} s".format(time() - t0))
    print("\n=========================================")

    if args.submit:
        test_data = pd.read_pickle(data_path / 'test_data.pkl')
        clf = KernelSVC(
            C=args.c,
            kernel=WesifeilerLehmanKernel(h_iter=args.h_iter,
                                          edges=args.edges)
        )
        clf.fit(Gn, labels)

        Gn_test = WL_preprocess(test_data)
        y_pred = clf.predict(Gn_test)
        pd.DataFrame(y_pred).to_csv('y_pred.csv', index=False)
