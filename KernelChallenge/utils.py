# %%
import networkx as nx

# %%


def graph_product(G1, G2):
    Gx = nx.Graph()
    for u, label1 in G1.nodes(data='labels'):
        for v, label2 in G2.nodes(data='labels'):
            if label1[0] == label2[0]:
                Gx.add_node((u, v), labels=label1)

    for u1, u2, label1 in G1.edges(data='labels'):
        for v1, v2, label2 in G2.edges(data='labels'):
            if ((u1, v1) in Gx) and (
                    (u2, v2) in Gx) and (
                    label1[0] == label2[0]):
                Gx.add_edge((u1, v1), (u2, v2), labels=label1)
            if ((u1, v2) in Gx) and (
                (u2, v1) in Gx) and (
                    label1[0] == label2[0]):
                Gx.add_edge((u1, v2), (u2, v1), labels=label1)
    return Gx
