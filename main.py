from argparse import ArgumentParser
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from KernelChallenge.kernels import WesifeilerLehmanKernel
from KernelChallenge.KLR import KernelLogisticRegression
from KernelChallenge.preprocessing import WL_preprocess
from KernelChallenge.SVC import KernelSVC


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/',
        help='Path to folder that contains the dataset (.pkl files)')
    parser.add_argument(
        '--n',
        type=int,
        default=1000,
        help='Number of samples from the dataset to use for training')
    parser.add_argument(
        '--h_iter',
        type=int,
        default=2,
        help='Number of iterations (depth) for the WL algorithm')
    parser.add_argument(
        '--c',
        type=float,
        default=1,
        help='Regularization parameter for the classifier')
    parser.add_argument(
        '--method',
        type=str,
        default="SVC",
        help='Classifier to use (SVC or KLR)')
    parser.add_argument(
        '--edges',
        action='store_true',
        help='Use edge embedding in the WL algorithm (see report for more '
        'details)')
    parser.add_argument(
        '--submit',
        action='store_true',
        help='create submission file for the challenge')
    return parser.parse_args()


def fit_predict_one(Gn, labels, train_index, test_index,
                    metric=roc_auc_score, c=100, h_iter=2, edges=False,
                    method="SVC"):
    X_train, X_test = Gn[train_index], Gn[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    WLK = WesifeilerLehmanKernel(h_iter=h_iter, edges=edges)

    if method == "SVC":
        clf = KernelSVC(C=c, kernel=WLK)

    elif method == "KLR":
        clf = KernelLogisticRegression(reg_param=c, kernel=WLK)

    predict_train = clf.fit(X_train, y_train)
    score_train = metric(y_train, predict_train)
    score_test = metric(y_test, clf.predict(X_test))
    print("AUC score on train set : {:.2f}".format(score_train))
    print("AUC score on test set : {:.2f}".format(score_test))
    return score_train, score_test


if __name__ == '__main__':
    args = parse_args()

    data_path = Path(args.data_path)
    train_data = pd.read_pickle(data_path / 'training_data.pkl')
    test_data = pd.read_pickle(data_path / 'training_data.pkl')
    labels = pd.read_pickle(data_path / 'training_labels.pkl')
    labels = 2 * labels - 1

    np.random.seed(0)
    idx = np.random.choice(len(train_data), args.n, replace=False)
    Gn = np.array(train_data, dtype=object)[idx]
    labels = labels[idx]

    Gn = WL_preprocess(Gn)

    N = len(Gn)
    t0 = time()
    scores = fit_predict_one(Gn,
                             labels,
                             np.arange(N * 4 // 5),
                             np.arange(N * 4 // 5, N),
                             c=args.c,
                             h_iter=args.h_iter,
                             edges=args.edges,
                             method=args.method)

    print("=========================================")
    print(f"C = {args.c}")
    print(
        f"\nAUC score on {args.n} samples using WL kernel and {args.method} : \
            {np.mean(scores[1])} ")
    print("Time : {:.2f} s".format(time() - t0))
    print("\n=========================================")

    if args.submit:
        test_data = pd.read_pickle(data_path / 'test_data.pkl')
        WLK = WesifeilerLehmanKernel(h_iter=args.h_iter,
                                     edges=args.edges)
        if args.method == "SVC":
            clf = KernelSVC(C=args.c, kernel=WLK)

        elif args.method == "KLR":
            clf = KernelLogisticRegression(reg_param=args.c, kernel=WLK)

        clf.fit(Gn, labels)

        Gn_test = WL_preprocess(test_data)
        y_pred = clf.predict(Gn_test)
        pd.DataFrame(y_pred).to_csv(
            f'y_pred_m_{args.method}_c_{args.c}_h_{args.h_iter}\
                _edges_{args.edges}_auc_tr_{round(scores[0],4)}\
                    _val_{round(scores[1],4)}.csv',
            index=False)
        print(y_pred[0:10])  # chceck for first prediction
