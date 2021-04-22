import time
import itertools

import numpy as np
import networkx as nx
from gsq.gsq_testdata import bin_data, dis_data
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import bnlearn
from pgmpy.factors.discrete import TabularCPD
import pgmpy
import pandas as pd

from utils import contained, flatten, cpdag_to_dags, generate_vals, parents, dag_topo_sort,generate_state_space,shared_contexts,data_to_contexts,context_per_stage
from cstree import dag_to_cstree1, cstree_pc1, stages_to_csi_rels1
from algorithms import dag_to_ci_model
from mincontexts import minimal_contexts, minimal_context_dags, binary_minimal_contexts1

"""
Plots for the thesis
"""

#========================================
# 3 DAG building blocks
#========================================
def dag_building_blocks_plot():
    options = {
                        'node_color': 'white',
                        'node_size': 1000,
        'arrowsize':20
                    }
    chainl = nx.DiGraph()
    chainl.add_edges_from([(1,2),(2,3)])
    plt.figure(figsize=(8,8/3))
    nx.draw_spectral(chainl, with_labels=True, **options)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000") 
    plt.savefig("figs/chainl.pdf")
    plt.show()
    chainr = nx.DiGraph()
    chainr.add_edges_from([(3,2),(2,1)])
    plt.figure(figsize=(8,8/3))
    nx.draw_spectral(chainr, with_labels=True, **options)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000") 
    plt.savefig("figs/chainr.pdf")
    plt.show()
    fork = nx.DiGraph()
    fork.add_edges_from([(2,1),(2,3)])
    plt.figure(figsize=(8,8/3))
    nx.draw_spectral(fork, with_labels=True, **options)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000") 
    plt.savefig("figs/fork.pdf")
    plt.show()
    collider = nx.DiGraph()
    collider.add_edges_from([(1,2),(3,2)])
    plt.figure(figsize=(8,8/3))
    nx.draw_spectral(collider, with_labels=True, **options)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000") 
    plt.savefig("figs/collider.pdf")
    plt.show()


#========================================
# DAG and non-DAG example
#========================================
def dag_and_non_dag_plot():
    options = {
                'node_color': 'white',
                'node_size': 1000,
                'arrowsize':20
                    }
    dag_eg = nx.DiGraph()
    dag_eg.add_edges_from([(1,2),(1,3),(2,4),(1,5)])
    plt.figure(figsize=(7,7))
    nx.draw_networkx(dag_eg,pos=nx.drawing.layout.spring_layout(dag_eg), **options)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000") 
    plt.savefig("figs/dageg.pdf")
    plt.show()


    # DAGs for Thesis
    options = {
                        'node_color': 'white',
                        'node_size': 1000,
        'arrowsize':20
                    }
    dag_neg = nx.DiGraph()
    dag_neg.add_edges_from([(1,2),(1,3),(2,4),(1,5),(4,6),(6,2)])
    plt.figure(figsize=(7,7))
    nx.draw_networkx(dag_neg,pos=nx.drawing.layout.spring_layout(dag_neg), **options)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000") 
    plt.savefig("figs/dagneg.pdf")
    plt.show()
    
    
#==========================================
# DAG To CSTree
#==========================================

def dag_to_cstree_plot():
    # CSI relatiosns straight from tree
    p3   = 5
    val_dict3 = {i+1:[0,1] for i in range(p3)}
    dag3 = nx.gnp_random_graph(p3,0.01,directed=True)
    dag3 = nx.DiGraph([(u,v) for (u,v) in dag3.edges() if u<v])
    dag3 = nx.relabel_nodes(dag3, lambda x: x +1)
    for i in range(1,p3+1):
        if i not in dag3.nodes:
            dag3.add_node(i)

    #dag3 = nx.DiGraph()
    #dag3.add_edges_from([(1,2),(2,3),(3,4),(4,5)])
    options = {
                        'node_color': 'white',
                        'node_size': 1000,
        'arrowsize':20
                    }
    tree3,stages3, cs3 = dag_to_cstree1(val_dict3, dag=dag3, ordering=[1,2,3,4,5])
    fig3, axes3 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    csirels = stages_to_csi_rels1(stages3, ordering=[1,2,3,4,5])
    csirels += decomposition(csirels)
    csirels += weak_union(csirels)
    pairwisecsirels = [rel for rel in csirels if len(rel[0].union(rel[1]))==2]




    print(pairwisecsirels)
    #axes3[0].set_title("DAG"),axes3[1].set_title("CSTree")
    l=nx.draw_networkx(dag3, ax=axes3[0],**options)

    node_colours3 = [cs3.get(n, "#FFFFFF") for n in tree3.nodes()]
    l=nx.draw_networkx(tree3,node_color = node_colours3, pos =  graphviz_layout(tree3, prog="dot", args=""), ax=axes3[1], with_labels=False)
    axes3[1].collections[0].set_edgecolor("#000000")
    axes3[0].collections[0].set_edgecolor("#000000")

    axes3[1].set_ylabel("$X_4$         $X_3$         $X_2$       $X_1$", fontsize=18)
    axes3[0].set_xlabel(r"$Random \ DAG\ G$", fontsize=18)
    axes3[1].set_xlabel(r"$CSTree $", fontsize=18)
    plt.savefig("figs/emptydagtocstree.pdf")
    
    
#==========================================
# DAG to CSTree and empty context DAG
#==========================================
def dag_to_cstree_mc_dag_plot():
# CSI relatiosns straight from tree
p3   = 5
val_dict3 = {i+1:[0,1] for i in range(p3)}
dag3 = nx.gnp_random_graph(p3,0.5,directed=True)
dag3 = nx.DiGraph([(u,v) for (u,v) in dag3.edges() if u<v])
dag3 = nx.relabel_nodes(dag3, lambda x: x +1)
for i in range(1,p3+1):
    if i not in dag3.nodes:
        dag3.add_node(i)
        
#dag3 = nx.DiGraph()
#dag3.add_edges_from([(1,2),(2,3),(3,4),(4,5)])
options = {
                    'node_color': 'white',
                    'node_size': 1000,
    'arrowsize':20
                }
tree3,stages3, cs3 = dag_to_cstree1(val_dict3, dag=dag3, ordering=[1,2,3,4,5])
fig3, axes3 = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

csirels = stages_to_csi_rels1(stages3, ordering=[1,2,3,4,5])
csirels += decomposition(csirels)
csirels += weak_union(csirels)
pairwisecsirels = [rel for rel in csirels if len(rel[0].union(rel[1]))==2]



mc_dag_dict = minimal_context_dags1([1,2,3,4,5], pairwisecsirels, val_dict3)
mc_dag = mc_dag_dict[0][1]
nx.draw_networkx(mc_dag, ax=axes3[2],**options)


print(pairwisecsirels)
#axes3[0].set_title("DAG"),axes3[1].set_title("CSTree")
l=nx.draw_networkx(dag3, ax=axes3[0],**options)

node_colours3 = [cs3.get(n, "#FFFFFF") for n in tree3.nodes()]
l=nx.draw_networkx(tree3,node_color = node_colours3, pos =  graphviz_layout(tree3, prog="dot", args=""), ax=axes3[1], with_labels=False)
axes3[1].collections[0].set_edgecolor("#000000")
axes3[2].collections[0].set_edgecolor("#000000")
axes3[0].collections[0].set_edgecolor("#000000")

axes3[1].set_ylabel("$X_4$         $X_3$         $X_2$       $X_1$", fontsize=18)
axes3[0].set_xlabel(r"$Random \ DAG\ G$", fontsize=18)
axes3[2].set_xlabel(r"$Empty \ context\ DAG\ from\ CSTree\ of\ G$", fontsize=18)
axes3[1].set_xlabel(r"$CSTree$", fontsize=18)
plt.savefig("figs/dagtocstreetoemptycontextdag.pdf")
    
#================================
# DAG to CSTree large p
#================================
# DAG to CSTree large p
def dag_to_cstree_mc_dag_large_p_plot():
    options = {
                        'node_color': 'white',
                        'node_size': 1000,
        'arrowsize':20
                    }
    save_figs = True
    p   = 10
    dag = nx.gnp_random_graph(p,0.7,directed=True)
    dag = nx.DiGraph([(u,v) for (u,v) in dag.edges() if u<v])
    dag = nx.relabel_nodes(dag, lambda x: x +1)
    for i in range(1,p+1):
        if i not in dag.nodes:
            dag.add_node(i)

    order = dag_topo_sort(dag)
    value_dict = {var:[0,1] for var in order}
    print("Chosen order is ",order)
    plt.figure(figsize=(7,7))
    nx.draw_networkx(dag,pos=nx.drawing.layout.shell_layout(dag,scale=4), **options)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    if save_figs:
        plt.savefig("figs/dagtocstree_dag"+str(p)+".pdf")

    t1=time.time()
    tree,stages,d = dag_to_cstree1(value_dict, order, dag)
    print(time.time()-t1)


    csi_rels        = []
    oldcsi_rels  = []
    for var in order[:-1]:
        stages_in_l = {k:v for k,v in stages.items() if nx.shortest_path_length(tree, "Root", v[0])==order.index(var)+1}


        stages_in_l = {k:v for k,v in stages.items() if len(v[0])==order.index(var)+1}

        for k,v in stages_in_l.items():


            common_context=context_per_stage(v)

            context_vars = [i for (i,j) in common_context]

            other_vars   = [i for (i,j) in v[0] if i not in context_vars]

            next_var = order[order.index(var)+1]
            #print("adding ",({next_var}, set(other_vars), set(), common_context))
            for o in other_vars:
            #    print(({next_var, o}, common_context))
                oldcsi_rels.append(({next_var, o}, common_context))


            csi_rels.append(({next_var}, set(other_vars), set(), common_context))
            #csi_rels   += csi_rels_from_stage
    pair_csi_rels = []
    for_min_contexts=[]
    for c in csi_rels:
        for var in c[1]:
            pair_csi_rels.append((c[0], {var}, c[1].difference({var}), c[3]))
            for_min_contexts.append((c[0].union({var}), c[1].difference({var}), c[3]))

    show_tree=True
    options = {
        'node_size': 500,
    }
    #print("Chosen order is ",order)
    if show_tree:
        fig=plt.figure(figsize=(12,12))
        pos =  graphviz_layout(tree, prog="twopi", args="")    
        #fig.suptitle("Causal ordering "+str(order) , fontsize=13)
        node_colours = [d.get(n, "#FFFFFF") for n in tree.nodes()]
        nx.draw(tree, node_color=node_colours, pos=pos, with_labels=False, font_color='white',linewidths=1, **options)
        ax  = plt.gca()
        ax.collections[0].set_edgecolor("#000000")
        if save_figs:
            plt.savefig("figs/dagtocstree_cstree"+str(p)+".pdf")

    tree_dseps = []
    for a in pair_csi_rels:
        if (a[0].union(a[1]), a[2]) not in tree_dseps:
            tree_dseps.append((a[0].union(a[1]), a[2]))

    mintemp = [({list(c[0])[0]}, {list(c[0])[1]},c[1],[]) for c in tree_dseps]

    all_mc_graphs = minimal_context_dags(order, oldcsi_rels, value_dict, mintemp)

    options = {
        'node_color': 'white',
        'node_size': 1000,
    }
    mecdag_count=0
    #print("d separations from original DAG")
    #print(dag_to_ci_model(dag),"\n")
    for mc,g in all_mc_graphs:


        #print("d separations from MEC DAG")
        #print(dag_to_ci_model(g),"\n")


        mecdag_count+=1
        fig=plt.figure(figsize=(7,7))
        context_string = "Context "+"".join(["X"+str(i)+"="+str(j)+" " for (i,j) in mc]) if mc != () else "Empty context DAG"
        #fig.suptitle(context_string, fontsize=13)
        nx.draw_networkx(g,pos=nx.drawing.layout.shell_layout(g,scale=4), **options)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")
        if save_figs:
            plt.savefig("figs/dagtocstree_mcdag"+str(p)+".pdf")



        print(g.edges==dag.edges)
        all_edges_in=True
        for e1 in list(g.edges):
            for e2 in list(dag.edges):
                all_edges_in = all_edges_in and (e1 in list(dag.edges) and e2 in list(g.edges))

        print("all edges in ", all_edges_in)

    plt.show()
    
    
    
#====================================
# DAG to CSTree to empty context DAG
#===================================


# Testing DAG to CSTree limits

def dag_to_cstree_to_emptycontext_dag(pedge=0.5):
    p   = 5
    dag = nx.gnp_random_graph(p,pedge,directed=True)
    dag = nx.DiGraph([(u,v) for (u,v) in dag.edges() if u<v])
    dag = nx.relabel_nodes(dag, lambda x: x +1)
    for i in range(1,p+1):
        if i not in dag.nodes:
            dag.add_node(i)

    order = dag_topo_sort(dag)
    value_dict = {var:[0,1] for var in order}
    print("Chosen order is ",order)


    order = dag_topo_sort(dag)
    value_dict = {var:[0,1] for var in order}
    plt.figure(figsize=(7,7))
    nx.draw_networkx(dag,pos=nx.drawing.layout.shell_layout(dag), **options)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000") 
    plt.savefig("figs/dagtocstree_dag"+str(p)+".pdf")

    print("Chosen order is ",order)
    t1=time.time()
    tree,stages,d = dag_to_cstree(value_dict, order, dag)
    print(time.time()-t1)

    if p<6:
        show_tree=True
    else:
        show_tree=False

    #print("d separations from original DAG")
    #print(dag_to_ci_model(dag),"\n")
    csi_rels        = []
    oldcsi_rels  = []
    for var in order[:-1]:
        stages_in_l = {k:v for k,v in stages.items() if nx.shortest_path_length(tree, "Root", v[0])==order.index(var)+1}


        stages_in_l = {k:v for k,v in stages.items() if len(v[0])==order.index(var)+1}

        for k,v in stages_in_l.items():


            common_context=context_per_stage(v)

            context_vars = [i for (i,j) in common_context]

            other_vars   = [i for (i,j) in v[0] if i not in context_vars]

            next_var = order[order.index(var)+1]
            #print("adding ",({next_var}, set(other_vars), set(), common_context))
            for o in other_vars:
            #    print(({next_var, o}, common_context))
                oldcsi_rels.append(({next_var, o}, common_context))


            csi_rels.append(({next_var}, set(other_vars), set(), common_context))
            #csi_rels   += csi_rels_from_stage
    pair_csi_rels = []
    for_min_contexts=[]
    for c in csi_rels:
        for var in c[1]:
            pair_csi_rels.append((c[0], {var}, c[1].difference({var}), c[3]))
            for_min_contexts.append((c[0].union({var}), c[1].difference({var}), c[3]))
       # csi_rels.remove(c)
    #print("Old style csi rels ", oldcsi_rels,"\n")
    #print("CSI Rels straight from Tree", csi_rels,"\n")
    #print("CSI Rels paired", pair_csi_rels,"\n")

    #print("d separations from original DAG")



    #print(dag_to_ci_model(dag),"\n")

    #print("d separations as per tree")
    """tree_dseps = []
    for a in pair_csi_rels:
        tree_dseps.append((a[0].union(a[1]), a[2]))"""
    #print(tree_dseps, "\n")


    #print("Chosen order is ",order)
    if show_tree:
        fig=plt.figure(figsize=(7,7))
        pos = graphviz_layout(tree, prog="dot", args='')
        #fig.suptitle("Causal ordering "+str(order) , fontsize=13)
        node_colours = [d.get(n, "#FFFFFF") for n in tree.nodes()]
        nx.draw(tree, node_color=node_colours, pos=pos, with_labels=False, font_color='white',linewidths=1)
        ax  = plt.gca()
        ax.collections[0].set_edgecolor("#000000")
        plt.savefig("figs/dagtocstree_cstree"+str(p)+".pdf")


    """
    dag_d_seps = dag_to_ci_model(dag)
    minimal_dseps=[]
    l = len(dag_d_seps)
    for i in range(l):
        d_sep = dag_d_seps[i]
        for j in range(l):
            if d_sep[1].issubset(dag_d_seps[j][1]) and d_sep[0]==dag_d_seps[j][0]:
                minimal_dseps.append((d_sep[0], d_sep[1]))
                break
    """
    #print("minimal dseps from dag \n" , minimal_dseps, "\n")

    #print("pair, d separations as per tree")
    tree_dseps = []
    for a in pair_csi_rels:
        if (a[0].union(a[1]), a[2]) not in tree_dseps:
            tree_dseps.append((a[0].union(a[1]), a[2]))
    #print(tree_dseps, "\n")

    #print("tree misses \n", [i for i in minimal_dseps if i not in tree_dseps])



    mintemp = [({list(c[0])[0]}, {list(c[0])[1]},c[1],[]) for c in tree_dseps]

    all_mc_graphs = minimal_context_dags(order, oldcsi_rels, value_dict, mintemp)

    #mcs=minimal_contexts(csi_rels, value_dict)
    #print(csi_rels)

    mecdag_count=0
    #print("d separations from original DAG")
    #print(dag_to_ci_model(dag),"\n")
    for mc,g in all_mc_graphs:


        #print("d separations from MEC DAG")
        #print(dag_to_ci_model(g),"\n")


        mecdag_count+=1
        fig=plt.figure(figsize=(7,7))
        context_string = "Context "+"".join(["X"+str(i)+"="+str(j)+" " for (i,j) in mc]) if mc != () else "Empty context DAG"
        #fig.suptitle(context_string, fontsize=13)
        nx.draw_networkx(g,pos=nx.drawing.layout.shell_layout(g), **options)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000") 
        plt.savefig("figs/dagtocstree_mcdag"+str(p)+".pdf")



        print(g.edges==dag.edges)
        all_edges_in=True
        for e1 in list(g.edges):
            for e2 in list(dag.edges):
                all_edges_in = all_edges_in and (e1 in list(dag.edges) and e2 in list(g.edges))

        print("all edges in ", all_edges_in)

    plt.show()
    #print("p nodes tree takes up ", (get_size(tree)/(1024**3)))
    #minimal_context_dict = minimal_contexts(oldcsi_rels, value_dict)
    #print("CI relations for empty context DAG")
    #print("Order is ", order)
    
    
#=========================
# Above circular graph
#===========================
# CSI relatiosns straight from tree
def dag_to_cstree_to_emptycontext_dag_circular():
    p3   = 5
    val_dict3 = {i+1:[0,1] for i in range(p3)}
    dag3 = nx.gnp_random_graph(p3,0.5,directed=True)
    dag3 = nx.DiGraph([(u,v) for (u,v) in dag3.edges() if u<v])
    dag3 = nx.relabel_nodes(dag3, lambda x: x +1)
    for i in range(1,p3+1):
        if i not in dag3.nodes:
            dag3.add_node(i)

    #dag3 = nx.DiGraph()
    #dag3.add_edges_from([(1,2),(2,3),(3,4),(4,5)])
    options = {
                        'node_color': 'white',
                        'node_size': 1000,
        'arrowsize':20
                    }
    tree3,stages3, cs3 = dag_to_cstree1(val_dict3, dag=dag3, ordering=[1,2,3,4,5])
    fig3, axes3 = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    csirels = stages_to_csi_rels1(stages3, ordering=[1,2,3,4,5])

    csirels += decomposition(csirels)
    csirels += weak_union(csirels)
    pairwisecsirels = [rel for rel in csirels if len(rel[0].union(rel[1]))==2]



    mc_dag_dict = minimal_context_dags1([1,2,3,4,5], pairwisecsirels, val_dict3)
    mc_dag = mc_dag_dict[0][1]
    nx.draw_networkx(mc_dag, ax=axes3[2],**options)

    mc_dag_edges = list(mc_dag.edges)
    random_dag_edges = list(dag3.edges)
    equal=True
    for e1 in mc_dag_edges:
        for e2 in random_dag_edges:
            equal = equal and (e1 in random_dag_edges) and (e2 in mc_dag_edges)
    print("both dags are equal ", equal)


    print(pairwisecsirels)
    #axes3[0].set_title("DAG"),axes3[1].set_title("CSTree")
    l=nx.draw_networkx(dag3, ax=axes3[0],**options)

    node_colours3 = [cs3.get(n, "#FFFFFF") for n in tree3.nodes()]
    l=nx.draw_networkx(tree3,node_color = node_colours3, pos =  graphviz_layout(tree3, prog="twopi", args=""), ax=axes3[1], with_labels=False)
    axes3[1].collections[0].set_edgecolor("#000000")
    axes3[2].collections[0].set_edgecolor("#000000")
    axes3[0].collections[0].set_edgecolor("#000000")

    axes3[1].set_ylabel("$X_4$         $X_3$         $X_2$       $X_1$", fontsize=18)
    axes3[0].set_xlabel(r"$Random \ DAG\ G$", fontsize=18)
    axes3[2].set_xlabel(r"$Empty \ context\ DAG\ from\ CSTree\ of\ G$", fontsize=18)
    axes3[1].set_xlabel(r"$CSTree$", fontsize=18)
    plt.savefig("figs/dagtocstreetoemptycontextdagtest.pdf")
    
    
    
#=======================================
# Staged tree CSTree difference
#=======================================

def stagedtree_cstree_plot():
# CSTRee and Staged tree plot

    p4   = 5
    val_dict4= {i+1:[0,1] for i in range(p4)}
    dag4 = nx.gnp_random_graph(p4,0.9,directed=True)
    dag4 = nx.DiGraph([(u,v) for (u,v) in dag4.edges() if u<v])
    dag4 = nx.relabel_nodes(dag4, lambda x: x +1)
    for i in range(1,p4+1):
        if i not in dag4.nodes:
            dag4.add_node(i)
    import matplotlib.pyplot as plt

    dag4.add_edges_from([(1,2),(2,3),(3,4),(4,5)])

    tree4,stages4, cs4 = dag_to_cstree1(val_dict4, dag=dag4, ordering=[1,2,3,4,5])
    fig4, axes4 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    csirels4 = stages_to_csi_rels1(stages4, ordering=[i+1 for i in range(p4)])
    csirels4 += decomposition(csirels4)
    csirels4 += weak_union(csirels4)
    pairwisecsirels = [rel for rel in csirels4 if len(rel[0].union(rel[1]))==2]
    print(pairwisecsirels)

    #axes4[0].set_title("DAG"),axes4[1].set_title(r"$CSTree$",)
    #nx.draw_networkx(dag4, ax=axes4[0])


    node_colours4 = [cs4.get(n, "#FFFFFF") for n in tree4.nodes()]
    node_colours4[4] = "#FF0000"
    node_colours4[3] = "#FF0000"

    node_colours4[7] = "#00FF00"
    node_colours4[9] = "#00FF00"

    node_colours4[10] = "#0000FF"  #green
    node_colours4[8]  = "#0000FF"
    node_colours4[12] = "#0000FF"
    node_colours4[14] = "#0000FF" #blue

    node_colours4[11] = "#FFFF00" #yelow
    node_colours4[13] = "#FFFF00"
    #node_colours4[5] = "#FF0000"
    nx.draw_networkx(tree4,node_color = node_colours4, pos =  graphviz_layout(tree4, prog="dot", args=""), ax=axes4[1], with_labels=False)

    #node_colours4[10] = "#FFFFFF"

    axes4[1].set_ylabel("$X_4$         $X_3$         $X_2$       $X_1$", fontsize=18)
    axes4[0].set_ylabel("$X_4$         $X_3$         $X_2$       $X_1$", fontsize=18)
    node_colours4[7] = "#0000FF"

    nx.draw_networkx(tree4,node_color = node_colours4, pos =  graphviz_layout(tree4, prog="dot", args=""), ax=axes4[0], with_labels=False)
    axes4[1].collections[0].set_edgecolor("#000000")
    axes4[0].collections[0].set_edgecolor("#000000")
    axes4[0].set_xlabel(r"$Staged\ tree \ model$", fontsize=18)
    axes4[1].set_xlabel(r"$CSTree$", fontsize=18)
    plt.savefig("figs/cstreestagedtree.pdf")
    
    
#============================================
# all orderings considered case plot
#===========================================

def all_orderings_case_plot():
        # CSI relatiosns straight from tree
    p3   = 5
    val_dict3 = {i+1:[0,1] for i in range(p3)}
    dag3 = nx.gnp_random_graph(p3,0.5,directed=True)
    dag3 = nx.DiGraph([(u,v) for (u,v) in dag3.edges() if u<v])
    dag3 = nx.relabel_nodes(dag3, lambda x: x +1)
    for i in range(1,p3+1):
        if i not in dag3.nodes:
            dag3.add_node(i)


    options = {
                        'node_color': 'white',
                        'node_size': 1000,
        'arrowsize':20
                    }
    tree3,stages3, cs3 = dag_to_cstree1(val_dict3, dag=dag3, ordering=[1,2,3,4,5])
    fig3, axes3 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    csirels = stages_to_csi_rels1(stages3, ordering=[1,2,3,4,5])
    csirels += decomposition(csirels)
    csirels += weak_union(csirels)
    pairwisecsirels = [rel for rel in csirels if len(rel[0].union(rel[1]))==2]




    mc_dag_dict = minimal_context_dags1([1,2,3,4,5], pairwisecsirels, val_dict3)
    mc_dag = mc_dag_dict[0][1]

    p3   = 4
    val_dict3 = {i+1:[0,1] for i in range(p3)}
    dag3 = nx.gnp_random_graph(p3,0.99,directed=True)
    dag3 = nx.DiGraph([(u,v) for (u,v) in dag3.edges() if u>v])
    dag3 = nx.relabel_nodes(dag3, lambda x: x +1)
    for i in range(1,p3+1):
        if i not in dag3.nodes:
            dag3.add_node(i)
    nx.draw_networkx(dag3, ax=axes3[1],**options)

    mc_dag_edges = list(mc_dag.edges)
    random_dag_edges = list(dag3.edges)
    equal=True
    for e1 in mc_dag_edges:
        for e2 in random_dag_edges:
            equal = equal and (e1 in random_dag_edges) and (e2 in mc_dag_edges)
    print("both dags are equal ", equal)


    dag3 = nx.DiGraph()
    dag3.add_edges_from([(1,2),(2,3)])
    l=nx.draw_networkx(dag3, ax=axes3[0],**options)

    node_colours3 = [cs3.get(n, "#FFFFFF") for n in tree3.nodes()]
    #l=nx.draw_networkx(tree3,node_color = node_colours3, pos =  graphviz_layout(tree3, prog="twopi", args=""), ax=axes3[1], with_labels=False)
    axes3[1].collections[0].set_edgecolor("#000000")
    axes3[0].collections[0].set_edgecolor("#000000")

    #axes3[1].set_ylabel("$X_4$         $X_3$         $X_2$       $X_1$", fontsize=18)
    axes3[0].set_xlabel(r"$Minimal\ Context\ DAG\ X_4=0$", fontsize=18)
    axes3[1].set_xlabel(r"$Empty \ context \ DAG$", fontsize=18)
    plt.savefig("figs/exampleonallmec.pdf")