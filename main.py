# %%
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from preprocessing import preprocess
from preprocessing import preprocess_one_dummy

# %%
data_path = Path('data/')
train_data = pd.read_pickle(data_path / 'training_data.pkl')
labels = pd.read_pickle(data_path / 'training_labels.pkl')
# %%
# %% Dummy classification using a SVC on atoms count vector

# %%
if __name__ == "__main__":
    preprocessed_graphs = preprocess(
        train_data, preprocess_f0=preprocess_one_dummy)
    labels = pd.read_pickle(data_path / 'training_labels.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_graphs, labels, stratify=labels, test_size=0.2)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("F1-score: {}\nAccuracy: {}".format(f1_score(y_pred, y_test),
          accuracy_score(y_pred, y_test)))
