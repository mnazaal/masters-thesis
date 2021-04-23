import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from cstree import dag_to_cstree1, stages_to_csi_rels1, decomposition, weak_union, cstree_pc1
from algorithms import dag_to_ci_model
from mincontexts import binary_minimal_contexts1,binary_minimal_contexts1,minimal_context_dags1
from utils import generate_vals


# testing after the cstree is weird for asia dataset
"""p3   = 5
val_dict3 = {i+1:[0,1] for i in range(p3)}
dag3 = nx.gnp_random_graph(p3,0.5,directed=True)
dag3 = nx.DiGraph([(u,v) for (u,v) in dag3.edges() if u<v])
dag3 = nx.relabel_nodes(dag3, lambda x: x +1)
for i in range(1,p3+1):
    if i not in dag3.nodes:
        dag3.add_node(i)"""
p=8
dag3 = nx.DiGraph()
val_dict3 = {i+1:[0,1] for i in range(p)}
nodes = [i+1 for i in range(p)]
ordering
dag3.add_edges_from([(1,2),(2,7), (6,3), (6,8)])
dag3.add_nodes_from(nodes)
options = {
                    'node_color': 'white',
                    'node_size': 1000,
    'arrowsize':20
                }
fig3, axes3 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig4, axes4 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

# change for magit
nx.draw_networkx(dag3, ax=axes3[0],**options)

ordering=[5, 4, 6, 3, 8, 1, 2, 7]

tree3,stages3, cs3 = dag_to_cstree1(val_dict3, dag=dag3, ordering=ordering)
node_colours3 = [cs3.get(n, "#FFFFFF") for n in tree3.nodes()]
nx.draw_networkx(tree3,node_color = node_colours3, pos =  graphviz_layout(tree3, prog="twopi", args=""), ax=axes4, with_labels=False)

# for cstree


csirels = stages_to_csi_rels1(stages3, ordering)
csirels += decomposition(csirels)
csirels += weak_union(csirels)
pairwisecsirels = [rel for rel in csirels if len(rel[0].union(rel[1]))==2]



mc_dag_dict = minimal_context_dags1([1,2,3,4,5], pairwisecsirels, val_dict3)
mc_dag = mc_dag_dict[0][1]
nx.draw_networkx(mc_dag, ax=axes3[1],**options)

print("order is ", ordering)

mc_dag_edges = list(mc_dag.edges)
random_dag_edges = list(dag3.edges)
equal=True
for e1 in mc_dag_edges:
    for e2 in random_dag_edges:
        equal = equal and (e1 in random_dag_edges) and (e2 in mc_dag_edges)
print("both dags are equal ", equal)


print("minimal contexts are", binary_minimal_contexts1(pairwisecsirels, val_dict3).keys())
#axes3[0].set_title("DAG"),axes3[1].set_title("CSTree")


axes4.collections[0].set_edgecolor("#000000")
axes3[1].collections[0].set_edgecolor("#000000")
axes3[0].collections[0].set_edgecolor("#000000")

axes4.set_ylabel("$X_4$         $X_3$         $X_2$       $X_1$", fontsize=18)
axes3[0].set_xlabel(r"$Random \ DAG\ G$", fontsize=18)
axes3[1].set_xlabel(r"$Empty \ context\ DAG\ from\ CSTree\ of\ G$", fontsize=18)
axes4.set_xlabel(r"$CSTree$", fontsize=18)
plt.savefig("figs/dagtocstreetoemptycontextdagtest.pdf")
