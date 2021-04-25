# Generating synthetic data from a random DAG
import pgmpy
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, MmhcEstimator

import time
import itertools

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


from utils import contained, flatten, cpdag_to_dags, generate_vals, parents, dag_topo_sort,generate_state_space,shared_contexts,data_to_contexts,context_per_stage,get_size
from cstree import  cstree_pc, stages_to_csi_rels
from datasets import synthetic_dag_binarydata, bnlearn_data,coronary_data
from utils import generate_dag, binary_dict, generate_state_space
from graphoid import graphoid_axioms
from mincontexts import minimal_context_dags, binary_minimal_contexts

def synthetic_dag_binarydata_experiment(nodes, p_edge, n):
    val_dict = binary_dict(nodes)
    dag = generate_dag(nodes, p_edge)
    dataset = synthetic_dag_binarydata(dag, n)

    list_stages, nonempty = cstree_pc(dataset)
    # Non empty minimal context DAGs
    for case in nonempty:
        (tree, stages, cs, ordering, mec_dag) = case
        
        node_colors = [cs.get(n, "#FFFFFF") for n in tree.nodes()]
        if nodes<6:
            pos = graphviz_layout(tree, prog="dot", args="")
        else:
            pos = graphviz_layout(tree, prog="twopi", args="")
        nx.draw(tree, node_color=node_colors, pos=pos, with_labels=False, font_color="white", linewidths=1)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")
        plt.show()

        csi_rels_from_tree = stages_to_csi_rels(stages.copy(),ordering)

        csi_rels = graphoid_axioms(csi_rels_from_tree.copy())

        minimal_contexts = binary_minimal_contexts(csi_rels.copy(), val_dict)

        all_mc_graphs = minimal_context_dags(ordering, csi_rels.copy(), val_dict)

        for mc,g in all_mc_graphs:
            fig, (ax1,ax2) = plt.subplots(1,2)
            options = {"node_color":"white","node_size":1000}
            ax1.set_title("MC DAG Context {}".format(mc))
            ax2.set_title("MEC DAG")
            nx.draw_networkx(g,pos = nx.drawing.layout.shell_layout(g), ax=ax1,**options)
            nx.draw_networkx(mec_dag, pos=nx.drawing.layout.shell_layout(mec_dag),ax=ax2,**options)
            ax1.collections[0].set_edgecolor("#000000")
            ax2.collections[0].set_edgecolor("#000000")
            plt.show()


    tree_stages = list_stages[0]
    (tree,stages,cs,ordering, mec_dag) = tree_stages
    node_colors = [cs.get(n, "#FFFFFF") for n in tree.nodes()]
    if nodes<6:
        pos = graphviz_layout(tree, prog="dot", args="")
    else:
        pos = graphviz_layout(tree, prog="twopi", args="")
    nx.draw(tree, node_color=node_colors, pos=pos, with_labels=False, font_color="white", linewidths=1)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    plt.show()

    csi_rels_from_tree = stages_to_csi_rels(stages.copy(),ordering)

    csi_rels = graphoid_axioms(csi_rels_from_tree.copy())

    minimal_contexts = binary_minimal_contexts(csi_rels.copy(), val_dict)

    all_mc_graphs = minimal_context_dags(ordering, csi_rels.copy(), val_dict)

    for mc,g in all_mc_graphs:
        fig, (ax1,ax2) = plt.subplots(1,2)
        options = {"node_color":"white","node_size":1000}
        ax1.set_title("MC DAG Context {}".format(mc))
        ax2.set_title("MEC DAG")
        nx.draw_networkx(g,pos = nx.drawing.layout.shell_layout(g), ax=ax1,**options)
        nx.draw_networkx(mec_dag, pos=nx.drawing.layout.shell_layout(mec_dag),ax=ax2,**options)
        ax1.collections[0].set_edgecolor("#000000")
        ax2.collections[0].set_edgecolor("#000000")
        plt.show()
    



def coronary_experiment():
    dataset = coronary_data()
    val_dict = generate_state_space(dataset)
    nodes = dataset.shape[1]
 
    list_stages, nonempty = cstree_pc(dataset, test="anderson",use_dag=False)
    
    # Non empty minimal context DAGs
    
    for case in nonempty:
        (tree, stages, cs, ordering, mec_dag) = case
        print("stages are\n")
        for c,ns in stages.items():
            print(c, len(ns) ,len(ns[0])) 
        csi_rels_from_tree = stages_to_csi_rels(stages.copy(),ordering)

        #print("from tree\n", csi_rels_from_tree)

        csi_rels = graphoid_axioms(csi_rels_from_tree.copy())

        #print("after axioms\n", csi_rels)
        
        minimal_contexts = binary_minimal_contexts(csi_rels.copy(), val_dict)

        #print("minimal contexts\n", minimal_contexts)

        #fig, ax = plt.subplots(2,num_mc_graphs)

        all_mc_graphs = minimal_context_dags(ordering, csi_rels.copy(), val_dict)
        num_mc_graphs = len(all_mc_graphs)
        

        fig=plt.figure(figsize=(24, 12))
        main_ax = fig.add_subplot(111)
        tree_ax = plt.subplot(2,1,2)
        ax = [plt.subplot(2, num_mc_graphs, i+1) for i in range(num_mc_graphs)]
        node_colors = [cs.get(n, "#FFFFFF") for n in tree.nodes()]
        if nodes<7:
            pos = graphviz_layout(tree, prog="dot", args="")
        else:
            pos = graphviz_layout(tree, prog="twopi", args="")
        nx.draw(tree, node_color=node_colors, ax=tree_ax,pos=pos, with_labels=False, font_color="white", linewidths=1)
        tree_ax.set_title("ordering is "+"".join(str(ordering)))
        tree_ax.set_ylabel("".join(str(ordering)))
        tree_ax.collections[0].set_edgecolor("#000000")
        #plt.show()


       
        for i, (mc,g) in enumerate(all_mc_graphs):
            options = {"node_color":"white","node_size":1000}
            ax[i].set_title("MC DAG Context {}".format(mc))
            nx.draw_networkx(g,pos = nx.drawing.layout.shell_layout(g), ax=ax[i],**options)
            ax[i].collections[0].set_edgecolor("#000000")

    plt.show()

        
    tree_stages = list_stages[0]
    (tree,stages,cs,ordering, mec_dag) = tree_stages
    node_colors = [cs.get(n, "#FFFFFF") for n in tree.nodes()]
    if nodes<6:
        pos = graphviz_layout(tree, prog="dot", args="")
    else:
        pos = graphviz_layout(tree, prog="twopi", args="")
    nx.draw(tree, node_color=node_colors, pos=pos, with_labels=False, font_color="white", linewidths=1)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    plt.show()

    csi_rels_from_tree = stages_to_csi_rels(stages.copy(),ordering)

    csi_rels = graphoid_axioms(csi_rels_from_tree.copy())

    minimal_contexts = binary_minimal_contexts(csi_rels.copy(), val_dict)

    all_mc_graphs = minimal_context_dags(ordering, csi_rels.copy(), val_dict)

    for mc,g in all_mc_graphs:
        fig, (ax1,ax2) = plt.subplots(1,2)
        options = {"node_color":"white","node_size":1000}
        ax1.set_title("MC DAG Context {}".format(mc))
        ax2.set_title("MEC DAG")
        nx.draw_networkx(g,pos = nx.drawing.layout.shell_layout(g), ax=ax1,**options)
        nx.draw_networkx(mec_dag, pos=nx.drawing.layout.shell_layout(mec_dag),ax=ax2,**options)
        ax1.collections[0].set_edgecolor("#000000")
        ax2.collections[0].set_edgecolor("#000000")
        plt.show()
    


coronary_experiment()
