# %%
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from preprocessing import preprocess
from preprocessing import preprocess_one

# %%
data_path = Path('data/')
train_data = pd.read_pickle(data_path / 'training_data.pkl')

# %%
# Try to color atoms with relevant colors (C: grey, O: red...)
nodes_colors = ['tab:red', 'tab:grey', 'tab:green', 'tab:orange',
                'tab:blue', 'tab:purple', 'tab:yellow', 'tab:cyan', 'tab:olive'
                ]
# Up to 50 atoms, add lots of colors for the rest
cm = plt.get_cmap('jet')(np.arange(40))
nodes_colors += list(cm)
# Same for bondings
edges_colors = ['k', 'tab:orange', 'tab:blue']


def plot_graph(G,
               ax=None,
               nodes_colors=nodes_colors,
               edges_colors=edges_colors,
               width=3,
               node_size=200):

    labels = nx.get_node_attributes(G, name="labels")
    labels_colors = {}
    edges_color_list = []
    for k, v in labels.items():
        labels[k] = v[0]
        labels_colors[k] = nodes_colors[v[0]]
    for _, _, bond in G.edges(data='labels'):
        edges_color_list.append(edges_colors[bond[0]])

    nx.draw(
        G,
        with_labels=True,
        labels=labels,
        font_weight='bold',
        node_color=list(labels_colors.values()),
        edge_color=edges_color_list,
        width=width,
        node_size=node_size,
        ax=ax,
    )


# %%

labels = pd.read_pickle(data_path / 'training_labels.pkl')
labels_pos = labels_pos = np.argwhere(labels).reshape(1, -1)[0]
fig, axes = plt.subplots(4, 4, figsize=(20, 20))

# Plot 16 molecules that have the poperty
for i in range(16):
    plot_graph(
        train_data[labels_pos[i]],
        ax=axes.ravel()[i])

# %% Need to preprocess the data in order to keep only the largest connected
# graph

preprocessed_graph = preprocess(
    train_data, preprocess_f0=preprocess_one, n_jobs=10)
fig, axes = plt.subplots(4, 4, figsize=(20, 20))

# Plot 16 molecules that have the poperty
for i in range(16):
    plot_graph(
        preprocessed_graph[labels_pos[i]],
        ax=axes.ravel()[i])  # %%

# %%
