# Generating synthetic data from a random DAG
import pgmpy
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, MmhcEstimator

import time
import itertools
import sys
import traceback
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from sacred import Experiment


from utils.utils import contained, flatten, cpdag_to_dags, generate_vals, parents, remove_cycles, dag_topo_sort,generate_state_space,shared_contexts,data_to_contexts,context_per_stage,get_size, binary_dict
from cstree import  cstree_pc, stages_to_csi_rels,  dag_to_cstree
from datasets import synthetic_dag_binarydata, bnlearn_data,coronary_data, dermatology_data, micecortex_data, susy_data, vitd_data
from utils.utils import generate_dag, binary_dict, generate_state_space
from graphoid import graphoid_axioms
from mincontexts import minimal_context_dags, binary_minimal_contexts
from models import CSTree


ex1 = Experiment("empty_context_dag")
ex2 = Experiment("synthetic_dag_binarydata")
ex3 = Experiment("dermatology")
ex4 = Experiment("micecortex")
ex5 = Experiment("susy")




#@ex1.automain
def dag_to_cstree_experiment(nodes,p_edge, mc_dag=False):
    dag = generate_dag(nodes,p_edge=p_edge)

    exp_name = str(nodes)+"-"+str(p_edge)

    val_dict = {i+1:[0,1] for i in range(nodes)}
    ordering = [i+1 for i in range(nodes)]

    fig=plt.figure(figsize=(5,5))
    options = {
    'node_color': 'white',
    'node_size': 1000,
                }

    nx.draw_networkx(dag,pos=nx.drawing.layout.shell_layout(dag), **options)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    plt.show()
    #plt.savefig("figs/dag_to_cstree/dag"+exp_name+".pdf")

    fig =plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    tree, stages, color_scheme, _,_ = dag_to_cstree(val_dict, ordering=ordering, dag=dag)
    tree_node_colors = [color_scheme.get(n, "#FFFFFF") for n in tree.nodes]
    except_last = [o for o in ordering[:-1]]
    cstree_ylabel = "".join(["     $X_{}$        ".format(o) for o in except_last[::-1]])
    tree_pos = graphviz_layout(tree, prog="dot", args="")
    ax.set_ylabel(cstree_ylabel)
    nx.draw_networkx(tree, node_color=tree_node_colors, ax=ax, pos=tree_pos,
                        with_labels=False, font_color="white", linewidths=1)
    ax.collections[0].set_edgecolor("#000000")
    plt.show()
    #plt.savefig("figs/dag_to_cstree/dag-cstree"+exp_name+".pdf")

    if mc_dag:
        csi_rels = stages_to_csi_rels(stages, ordering)
        csi_rels = graphoid_axioms(csi_rels, val_dict, specialize=False)
        mc_dags = minimal_context_dags(ordering, csi_rels, val_dict)
        for i, (mc, mc_dag) in enumerate(mc_dags):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            options = {
            'node_color': 'white',
            'node_size': 1000,
                        }
            nx.draw_networkx(mc_dag, pos=nx.drawing.layout.shell_layout(mc_dag),ax=ax1,**options)
            #ax1.set_title(mcdag_title)
            #ax1.collections.set_edgecolor("#000000")
            nx.draw_networkx(dag,pos=nx.drawing.layout.shell_layout(dag),ax=ax2, **options)
            #ax2.collections.set_edgecolor("#000000")
            plt.show()
        #plt.savefig("figs/dag_to_cstree/mc-dag"+str(i)+exp_name+".pdf")



def synthetic_dag_binary_data_experiment(dataset):


    fig=plt.figure(figsize=(5,5))
    options = {
    'node_color': 'white',
    'node_size': 1000,
                }

    #nx.draw_networkx(dag,pos=nx.drawing.layout.shell_layout(dag), **options)
    #ax = plt.gca()
    #ax.collections[0].set_edgecolor("#000000")
    #plt.savefig("figs/synthetic/dag"+exp_name+".pdf")


    n,p = dataset.shape
    ordering=[i+1 for i in range(p)]
    val_dict = {i+1:[0,1] for i in range(p)}



    cstree_object=CSTree(dataset,val_dict)
    cstree_object.visualize(orderings=[ordering], csi_test="epps", return_type="all", plot_mcdags=True,use_dag=True, learn_limit=None, plot_limit=None)

#@ex2.config
def synthetic_dag_binarydata_config():
    nodes=6
    p_edge=1
    n=10000
    use_dag=False
    csi_test="anderson"
    balanced_samples_only=True

#@ex2.automain
def synthetic_dag_binarydata_experiment_1(dataset, use_dag):
    # Increasing sample sizes until we get the empty context

    n,p = dataset.shape
    val_dict = {i+1:[0,1] for i in range(p)}


    for i in range(1):
        #dataset_subsample_size = int(n/(10**(i)))
        #dataset_subsample_indices = random.sample(range(n), dataset_subsample_size)
        dataset_subsample_indices = tuple(j for j in range(int(n/(10**i))))
        dataset_subsample         = dataset[tuple(dataset_subsample_indices),:]
        cstree_object = CSTree(dataset_subsample, val_dict)
        ordering=[i+1 for i in range(p)]
        #ordering=None
    #assert set(ordering)==set(list(val_dict.keys()))
        try:
            ordering_count = cstree_object.count_orderings()
            print("Current experiment has {} orderings".format(ordering_count))
            #cstree_object.visualize(plot_mcdags=True,csi_test="anderson",ordering=ordering, use_dag=use_dag, save_dir=exp_name+str(i))
            cstree_object.learn(ordering=[1, 2, 3, 4, 5, 6],get_bic=True,csi_test="anderson",all_trees=True, use_dag=True)
        except ValueError as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno
            print("WARNING: Exception in synthetic DAGs experiment")
            print("{} in file {} line {} with message \"{}\"".format(str(exception_type)[7:-1], filename, line_number, e))
            print("Full traceback is")
            for line in (traceback.format_tb(exception_traceback)):
                print(line)






@ex3.config
def dermatology_config():
    grouped=True
    csi_test="anderson"

#@ex3.automain
def dermatology_experiment():
    # Anderson test
    # with and without DAG
    dataset = dermatology_data(grouped)
    nodes = dataset.shape[1]
    val_dict = {i+1:[0,1] for i in range(nodes)}
    cstree_object = CSTree(dataset, val_dict)

    #cstree_object.learn(all_trees=False)

    cstree_object.visualize(use_dag=False)


@ex4.config
def micecortex_config():
    num_features=10
    
#@ex4.automain
def micecortex_experiment(num_features, cpdag_method, csi_test, return_type, visualize=False, plot_mcdags=False):
    # Anderson test
    # With and without CI relations
    # With 6 features and 10 features
    kl_threshold=None
    if csi_test not in ["anderson", "epps"]:
        assert isinstance(csi_test, float)
        kl_threshold=csi_test
        csi_test="kl"

    dataset = micecortex_data(num_features)
    n,p=dataset.shape


    val_dict = {i+1:[0,1] for i in range(p)}
    # Last variable, which is the predictor, has 8 possible classes
    val_dict[p]=[i for i in range(8)]

    cstree_object = CSTree(dataset,val_dict)

    

    if not visualize:
        cstree_object.learn(cpdag_method=cpdag_method,return_type=return_type,csi_test=csi_test, kl_threshold=kl_threshold,learn_limit=None)
    else:
        cstree_object.visualize(plot_mcdags=plot_mcdags,cpdag_method=cpdag_method,return_type=return_type,csi_test=csi_test, kl_threshold=kl_threshold,learn_limit=None,plot_limit=None)


#micecortex_experiment(6)

@ex5.config
def susy_config():
    ratio=0.1
    csi_test="anderson"
    use_dag=True
    balanced_samples_only=True
    #make save_dir here
    

#@ex5.automain
def susy_experiment(cpdag_method, csi_test, return_type, ratio):
    # remove_var: Variable to remove
    # cpdag_method: 
    kl_threshold=None
    if csi_test not in ["anderson", "epps"]:
        assert isinstance(csi_test, float)
        kl_threshold=csi_test
        csi_test="kl"

    dataset = susy_data(False, ratio)

    n,p = dataset.shape

    val_dict = {i+1:[0,1] for i in range(p)}

    cstree_object = CSTree(dataset, val_dict) 

    cstree_object.visualize(cpdag_method=cpdag_method,return_type=return_type,csi_test=csi_test, kl_threshold=kl_threshold,learn_limit=1)


    
def vitd_experiment(cpdag_method, csi_test, return_type, remove_vars=None, plot_limit=None, visualize=False, plot_mcdags=False):
    # remove_var: Variable to remove
    # cpdag_method: 
    kl_threshold=None
    if csi_test not in ["anderson", "epps"]:
        assert isinstance(csi_test, float)
        kl_threshold=csi_test
        csi_test="kl"
        
    # Load dataset
    dataset = vitd_data()
    
    #if remove_vars:
    #dataset = dataset[:,tuple(i for i in range(dataset.shape[1]) if i!=1)]
    
    n,p = dataset.shape

        
    val_dict = {}
    val_dict[1]=[0,1,2,3]
    val_dict[2]=[0,1]
    val_dict[3]=[0,1,2,3]
    val_dict[4]=[0,1]
    val_dict[5]=[0,1]
    
    # Create CSTree object
    cstree_object = CSTree(dataset, val_dict)
    
    # Learn the cstree and output the best BIC values of the following cases:
    # 1. Just the DAG converted to a CSTree
    # 2. CSTree learnt without any CI relations from DAG
    # 3. CSTree learnt using CI relations from DAG

    # our fixed orderings, rm
    orderings = list(itertools.permutations([1, 2, 3]))
    orderings = [list(o)+[4,5] for o in orderings]
    o1 = [3,2,1,4,5]
    o2 = [5,3,4,2,1]

    # to visualize mcdags which is somehow tractable here
    if not visualize:
        cstree_object.learn(cpdag_method=cpdag_method,return_type=return_type,csi_test=csi_test, kl_threshold=kl_threshold,learn_limit=None)
    else:
        cstree_object.visualize(orderings=[o2],plot_mcdags=plot_mcdags,cpdag_method=cpdag_method,return_type=return_type,csi_test=csi_test, kl_threshold=kl_threshold,learn_limit=None,plot_limit=None)

    
def coronary_experiment(cpdag_method, csi_test, return_type, remove_vars=None, plot_limit=None, visualize=False, plot_mcdags=False):
    # remove_var: Variable to remove
    # cpdag_method: 
    kl_threshold=None
    if csi_test not in ["anderson", "epps"]:
        assert isinstance(csi_test, float)
        kl_threshold=csi_test
        csi_test="kl"
        
    # Load dataset
    dataset = coronary_data()
    
    
    if cpdag_method=="all":
        # Can only try all DAGs on 5 variables efficiently
        assert remove_vars is not None
    if remove_vars:
        dataset = dataset[:,tuple(i for i in range(dataset.shape[1]) if i+1 not in remove_vars)]
    
    n,p = dataset.shape
        
    val_dict = {i+1:[0,1] for i in range(p)}
    
    # Create CSTree object
    
    cstree_object = CSTree(dataset, val_dict)
    
    # Learn the cstree and output the best BIC values of the following cases:
    # 1. Just the DAG converted to a CSTree
    # 2. CSTree learnt without any CI relations from DAG
    # 3. CSTree learnt using CI relations from DAG
    if not visualize:
        cstree_object.learn(cpdag_method=cpdag_method,return_type=return_type,csi_test=csi_test, kl_threshold=kl_threshold,learn_limit=None)
    else:
        cstree_object.visualize(plot_mcdags=plot_mcdags,cpdag_method=cpdag_method,return_type=return_type,csi_test=csi_test, kl_threshold=kl_threshold,learn_limit=None,plot_limit=None)
                        #save_dir=cpdag_method+csi_test+"_coronary_bic")

def coronary_experiment_mcdag(p):
    # remove_var: Variable to remove
    # cpdag_method: 

    for i in [1]:
        removing_vars = []
        dataset = coronary_data()
        dataset = dataset[:, tuple(i for i in range(dataset.shape[1]) if i+1 not in removing_vars)]
        n,p=dataset.shape
        val_dict = {i+1:[0,1] for i in range(p)}
        cstree_object=CSTree(dataset,val_dict)
        cstree_object.visualize(plot_mcdags=True,cpdag_method="hill",return_type="maxbic",csi_test="kl", kl_threshold=5e-7,learn_limit=None)

    
    # Learn the cstree and output the best BIC values of the following cases:
    # 1. Just the DAG converted to a CSTree
    # 2. CSTree learnt without any CI relations from DAG
    # 3. CSTree learnt using CI relations from DAG
    # cstree_object.learn(cpdag_method=cpdag_method,return_type=return_type,csi_test=csi_test, kl_threshold=kl_threshold,learn_limit=None)
#dag_to_cstree_experiment(5,1)
#dag_to_cstree_experiment(5,0)
#dag_to_cstree_experiment(5,0.5, mc_dag=True)
"""
nodes=4
p_edge=0.7
samples=10000
dag = generate_dag(nodes,p_edge=p_edge)
exp_name = str(nodes)+"-"+str(p_edge)+"-"+str(samples)
dataset = synthetic_dag_binarydata(dag, samples)
synthetic_dag_binary_data_experiment(dataset,dag,exp_name,use_dag=False)
dataset = dataset[tuple(random.sample(range(dataset.shape[0]), int(samples/10))),:]
synthetic_dag_binary_data_experiment(dataset,dag,exp_name,use_dag=False)
"""
# coronary experiments
# Minimum stages
# CPDAG method, staging method change
"""
print("\n\n!!!!!Coronary experiment!!!!\n\n")
for objective in ["minstages", "maxbic"]:
    for cpdag_method in ["hill"]:
        for merge_method in ["anderson", "epps", 5e-5,5e-6,5e-7]:
            print("\n!!!!!!!!!!!!!!!! coronary Starting on config {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!".format(cpdag_method, merge_method, objective))
            coronary_experiment(cpdag_method, merge_method, objective)
            print("!!!!!!!!!!!!!!!! End on config {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!\n".format(cpdag_method, merge_method, objective))

print("\n\n!!!!!Mice cortex experiment!!!!\n\n")
for objective in ["minstages", "maxbic"]:
    for cpdag_method in ["hill"]:
        for merge_method in ["anderson", "epps", 5e-4,5e-5,5e-6]:
            print("\n!!!!!!!!!!!!!!!! Mice Starting on config {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!".format(cpdag_method, merge_method, objective))
            micecortex_experiment(7,cpdag_method, merge_method, objective)
            print("!!!!!!!!!!!!!!!! End on config {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!\n".format(cpdag_method, merge_method, objective))

print("\n\n!!!!!VitaminD experiment!!!!\n\n")
for objective in ["minstages", "maxbic"]:
    for cpdag_method in ["pc1", "hill"]:
        for merge_method in ["anderson", "epps", 5e-1,5e-2,5e-4]:
            print("\n !!!!!!!!!!!!!!!! vitamind Starting on config {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!".format(cpdag_method, merge_method, objective))
            vitd_experiment(cpdag_method, merge_method, objective)
            print("!!!!!!!!!!!!!!!! End on config {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!\n".format(cpdag_method, merge_method, objective))
"""

#coronary_experiment("hill", 5e-7, "maxbic", visualize=True, plot_mcdags=True)
# minimum stage tree
#coronary_experiment("pc1", "epps", "minstages")
#coronary_experiment("hill", 5e-7, "maxbic", visualize=True)
#micecortex_experiment(7, "pc1", "epps", "minstages", visualize=True)
#vitd_experiment("pc1", 5e-2, "maxbic", visualize=True)
#for remove_var in [1,2,3,5]:
#coronary_experiment("pc1", 5e-5, "maxbic", visualize=True, plot_mcdags=True, remove_vars=[4])
#micecortex_experiment(7, "pc1", "epps", "minstages", visualize=True)
vitd_experiment("pc1", 5e-3, "minstages", visualize=True, plot_mcdags=True)
#micecortex_experiment(4, "pc1", 5e-5, "minstages", visualize=True, plot_mcdags=True)
#dag_to_cstree_experiment(5,0.5, mc_dag=True)

#dag = generate_dag(6, 0.5)
#dag=nx.DiGraph()
#dag.add_edges_from([(1,4),(2,4),(3,4)])
#print("True DAG edges",list(dag.edges))
#dataset = synthetic_dag_binarydata(dag, 100000)
#np.savetxt('bic_dataset.csv', dataset, fmt="%d", delimiter=",")
#dataset=np.genfromtxt('bic_dataset2.csv', delimiter=',').astype(int)
# TODO Save this
#synthetic_dag_binary_data_experiment(dataset)
#samples =random.sample(range(2000), 500)
#dataset1 = dataset[tuple(samples),:].copy()
#np.savetxt('bic_dataset3.csv', dataset1, fmt="%d", delimiter=",")
#dataset1=np.genfromtxt('bic_dataset3.csv', delimiter=',').astype(int)
#synthetic_dag_binary_data_experiment(dataset1)
#synthetic_dag_binarydata_experiment_1(dataset, use_dag=False)