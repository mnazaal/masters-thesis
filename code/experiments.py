# Generating synthetic data from a random DAG
import pgmpy
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, MmhcEstimator

import time
import itertools
import sys
import traceback

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from sacred import Experiment


from utils.utils import contained, flatten, cpdag_to_dags, generate_vals, parents, dag_topo_sort,generate_state_space,shared_contexts,data_to_contexts,context_per_stage,get_size, binary_dict
from cstree import  cstree_pc, stages_to_csi_rels
from datasets import synthetic_dag_binarydata, bnlearn_data,coronary_data, dermatology_data, micecortex_data, susy_data
from utils.utils import generate_dag, binary_dict, generate_state_space
from graphoid import graphoid_axioms
from mincontexts import minimal_context_dags, binary_minimal_contexts
from models import CSTree


ex1 = Experiment("empty_context_dag")
ex2 = Experiment("synthetic_dag_binarydata")
ex3 = Experiment("dermatology")
ex4 = Experiment("micecortex")
ex5 = Experiment("susy")

@ex1.config
def empty_context_dag_config():
    nodes=10

@ex1.automain
def empty_context_dag_experiment(nodes):
    full_dag = generate_dag(nodes,p_edge=1)
    empty_dag = generate_dag(nodes, p_edge=0)

    
    # Implement function to assert recovered empty context
    # DAGs are same for fully connected case and empty case
    pass


@ex2.config
def synthetic_dag_binarydata_config():
    nodes=6
    p_edge=1
    n=10000
    use_dag=False
    csi_test="anderson"
    balanced_samples_only=True

@ex2.automain
def synthetic_dag_binarydata_experiment(nodes, p_edge,n,use_dag,csi_test,balanced_samples_only):
    # Using epps and Anderson
    # With and without DAG CI relations
    # Effect of unbalanced data with Anderson test on full DAG
    dag = generate_dag(nodes, p_edge)
    dataset = synthetic_dag_binarydata(dag, n)
    val_dict = {i+1:[0,1] for i in range(nodes)}
    cstree_object = CSTree(dataset, val_dict)
    #ordering=[i+1 for i in range(nodes)]
    ordering=None
    #assert set(ordering)==set(list(val_dict.keys()))
    try:
        ordering_count = cstree_object.count_orderings()
        print("Current experiment has {} orderings".format(ordering_count))
        cstree_object.visualize(ordering=ordering, use_dag=use_dag)
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
def micecortex_experiment():
    dataset = micecortex_data(num_features)

    cstree_object = CSTree(dataset)

    cstree_object.visualize(all_trees=False, plot_limit=None, learn_limit=None)

@ex5.config
def susy_config():
    ratio=0.1
    csi_test="anderson"
    use_dag=True
    balanced_samples_only=True
    #make save_dir here
    

#@ex5.automain
def susy_experiment():
    dataset = susy_data(False, ratio)

    n,p = dataset.shape

    val_dict = {i+1:[0,1] for i in range(p)}

    cstree_object = CSTree(dataset, val_dict) 

    cstree_object.visualize(all_trees=True, save_dir=save_dir, use_dag=False, plot_limit=10, learn_limit=10)

    

    
def coronary_experiment():
    # Load dataset
    dataset = coronary_data()
    
    # Create CSTree object
    cstree_object = CSTree(dataset)

    # Visualize CSTrees with fewest stages
    save_dir=None
    cstree_object.visualize(all_trees=False, plot_limit=None, learn_limit=None)

#synthetic_dag_binarydata_experiment(9, 0.2, 350)
#coronary_experiment()
#dermatology_experiment(grouped=True)
#micecortex_experiment()
#susy_experiment()

