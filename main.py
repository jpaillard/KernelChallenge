from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import random
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
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
    parser.add_argument('--edges', action='store_true')
    parser.add_argument('--submit', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_path = Path(args.data_path)
    train_data = pd.read_pickle(data_path / 'training_data.pkl')
    test_data = pd.read_pickle(data_path / 'training_data.pkl')
    labels = pd.read_pickle(data_path / 'training_labels.pkl')  
    
    # Gn = WL_preprocess(random.sample(train_data, args.n))
    Gn = WL_preprocess(train_data[:args.n])
    print(len(Gn))
    X_train, X_test, y_train, y_test = train_test_split(
        Gn,
        labels[:args.n],
        test_size=0.2,
        random_state=0,
        stratify=labels[:args.n], shuffle = True
    )

    WLK = WesifeilerLehmanKernel(h_iter=args.h_iter, edges=args.edges)
    feat_train = WLK.fit_subtree(X_train)
    feat_test = WLK.predict(X_test)

    K_train = gramMatrix(feat_train, feat_train)
    K_test = gramMatrix(feat_test, feat_train)
    clf = SVC(C=args.c, kernel='precomputed')
    clf.fit(K_train, y_train)



    print("=========================================")
    print("\nF1 score on {:d} samples using WL kernel :  {:.2f}".format(
        args.n, f1_score(y_test, clf.predict(K_test))))
    print("\nAccuracy score on {:d} samples using WL kernel :  {:.2f}".format(
        args.n, accuracy_score(y_test, clf.predict(K_test))))
    print("\nRecall score on {:d} samples using WL kernel :  {:.2f}".format(
        args.n, recall_score(y_test, clf.predict(K_test))))
    print("\nPrecision score on {:d} samples using WL kernel :  {:.2f}".format(
        args.n, precision_score(y_test, clf.predict(K_test))))
    print("\n=========================================")



    # TODO  : Retrain on the whole training set using CValidated parameters

    if args.submit:
        Gn = WL_preprocess(test_data)
        feat_test = WLK.predict(test_data)
        K_test = gramMatrix(feat_test, feat_train)
        y_pred = clf.predict(K_test)
        pd.DataFrame(y_pred).to_csv('y_pred.csv', index=False)