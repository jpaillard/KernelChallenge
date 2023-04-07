# %%
import pandas as pd

# %%
# renommer l'output y_pred_c c: la valeur du param utilisé (éviter
# d'écraser les fichiers)
c = 1
df = pd.read_csv(f'y_pred_{str(c)}.csv', names=['Predicted'])
df.drop(0, inplace=True)
df.index.name = 'Id'
df.head()
# %%
test_data = pd.read_pickle('data/test_data.pkl')
assert len(df.Predicted) == len(test_data)
# %%
df.to_csv(f'submission_{str(c)}.csv')
# %%
