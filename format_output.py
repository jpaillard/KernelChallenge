# %%
import pandas as pd

# %%
# renommer l'output y_pred_c{val}_h{val} c: la valeur du param utilisé
# (éviter d'écraser les fichiers)
c = 0.01
h = 3
df = pd.read_csv(f'y_pred_c{str(c)}_h{str(h)}.csv', names=['Predicted'])
df.drop(0, inplace=True)
df.index.name = 'Id'
df.head()
# %%
test_data = pd.read_pickle('data/test_data.pkl')
assert len(df.Predicted) == len(test_data)
# %%
df.to_csv(f'submission_c{str(c)}_h{str(h)}.csv')
# %%
