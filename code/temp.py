import matplotlib.pyplot as plt
import networkx as nx

coronary_pc_cpdag = nx.DiGraph()
coronary_pc_cpdag.add_edges_from([(1, 2), (2, 5), (3, 1), (3, 2), (4, 2), (4, 5), (4, 1)])
coronary_pc_cpdag.add_nodes_from([i+1 for i in range(6)])
coronary_hc_cpdag = nx.DiGraph()
coronary_hc_cpdag.add_edges_from([(1, 3), (1, 5), (1, 4), (1, 2), (2, 6), (3, 1), (3, 5), (3, 2), (4, 5), (4, 1), (4, 2), (5, 3), (5, 1), (5, 4), (5, 2)])
coronary_hc_cpdag.add_nodes_from([i+1 for i in range(6)])
mice_pc_cpdag = nx.DiGraph()
mice_pc_cpdag.add_edges_from([(1, 8), (2, 1), (2, 5), (2, 8), (3, 8), (3, 6), (3, 5), (4, 8), (4, 7), (5, 8), (6, 5), (6, 8), (7, 6), (7, 8), (7, 5), (7, 4)])
mice_pc_cpdag.add_nodes_from([i+1 for i in range(8)])
mice_hc_cpdag = nx.DiGraph()
mice_hc_cpdag.add_edges_from( [(1, 8), (1, 3), (1, 2), (1, 5), (1, 4), (1, 7), (2, 1), (2, 8), (2, 5), (2, 7), (3, 8), (3, 1), (3, 5), (3, 7), (3, 6), (4, 7), (4, 6), (5, 4), (5, 7), (5, 6), (7, 6), (8, 3), (8, 1), (8, 2), (8, 5), (8, 4), (8, 7), (8, 6)])
mice_hc_cpdag.add_nodes_from([i+1 for i in range(8)])
vitd_pc_cpdag = nx.DiGraph()
vitd_pc_cpdag.add_edges_from([(1, 5), (3, 4), (5, 4), (5, 1)])
vitd_pc_cpdag.add_nodes_from([i+1 for i in range(5)])
vitd_hc_cpdag = nx.DiGraph()
vitd_hc_cpdag.add_edges_from([(5, 4), (3, 4), (3, 5), (1, 5)])
vitd_hc_cpdag.add_nodes_from([i+1 for i in range(5)])
cpdags = [coronary_pc_cpdag,coronary_hc_cpdag,mice_pc_cpdag,mice_hc_cpdag,vitd_pc_cpdag,vitd_hc_cpdag]
cpdags = [coronary_pc_cpdag,coronary_hc_cpdag,mice_pc_cpdag,mice_hc_cpdag]

synthetic_dag = nx.DiGraph()
synthetic_dag.add_edges_from([(1, 3), (1, 5), (1, 4), (1, 2), (2, 6), (3, 1), (3, 5), (3, 2), (4, 5), (4, 1), (4, 2), (5, 3), (5, 1), (5, 4), (5, 2)])
#cpdags = [synthetic_dag]

options = {"node_color":"white", "node_size":1000}
for i,cpdag in enumerate(cpdags):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    if i%2==0:
        cpdag_pos = nx.drawing.layout.shell_layout(cpdag)
    nx.draw_networkx(cpdag, pos=cpdag_pos, arrowsize=20, **options)
    ax.collections[0].set_edgecolor("#000000")
    plt.show()