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

#@ex1.automain
def empty_context_dag_experiment(nodes):
    full_dag = generate_dag(nodes,p_edge=1)
    empty_dag = generate_dag(nodes, p_edge=0)
    
    # Implement function to assert recovered empty context
    # DAGs are same for fully connected case and empty case
    pass


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

def synthetic_dag_binarydata_experiment_2(nodes, p_edge, n):
    # Using epps and Anderson
    pass

def synthetic_dag_binarydata_experiment_3(nodes, p_edge, n):
    # With and without CI relations from DAG
    pass

def synthetic_dag_binarydata_experiment_4(nodes, p_edge, n):
    # Effect of not including imbalanced data
    pass


#dag = generate_dag(6, 0.5)
#print("True DAG edges",list(dag.edges))
#dataset = synthetic_dag_binarydata(dag, 100000)
#np.savetxt('bic_dataset.csv', dataset, fmt="%d", delimiter=",")
#dataset=np.genfromtxt('bic_dataset.csv', delimiter=',').astype(int)
# TODO Save this
#synthetic_dag_binarydata_experiment_1(dataset, use_dag=True)
#synthetic_dag_binarydata_experiment_1(dataset, use_dag=False)


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
def micecortex_experiment(num_features):
    # Anderson test
    # With and without CI relations
    # With 6 features and 10 features
    dataset = micecortex_data(num_features)
    n,p=dataset.shape


    val_dict = {i+1:[0,1] for i in range(p)}
    val_dict[7]=[i for i in range(8)]

    cstree_object = CSTree(dataset,val_dict)

    

    cstree_object.visualize(all_trees=False, use_dag=False,plot_limit=None, learn_limit=None, save_dir="mice6_nodag")


#micecortex_experiment(6)

@ex5.config
def susy_config():
    ratio=0.1
    csi_test="anderson"
    use_dag=True
    balanced_samples_only=True
    #make save_dir here
    

#@ex5.automain
def susy_experiment(ratio):
    dataset = susy_data(True, ratio)

    n,p = dataset.shape

    val_dict = {i+1:[0,1] for i in range(p)}

    cstree_object = CSTree(dataset, val_dict) 

    cstree_object.visualize(all_trees=True, csi_test="epps",save_dir="susy", use_dag=False, plot_limit=5, learn_limit=5)

    

    
def coronary_experiment():
    # Load dataset
    dataset = coronary_data()[:,:-1]
    n,p = dataset.shape
    val_dict = {i+1:[0,1] for i in range(p)}
    
    # Create CSTree object
    cstree_object = CSTree(dataset, val_dict)
    

    # Visualize CSTrees with fewest stages
    save_dir=None
    cstree_object.learn(cpdag_method="hill",get_bic=True,csi_test="kl",all_trees=True, use_dag=True, learn_limit=None)


def coronary_experiment_bic():
    # Load dataset
    dataset = coronary_data()
    n,p=dataset.shape
    val_dict = {i+1:[0,1] for i in range(p)}
    
    # Create CSTree object
    cstree_object = CSTree(dataset, val_dict)
    

    # Visualize CSTrees with fewest stages
    save_dir=None
    trees = cstree_object.learn(cpdag_method="hill",get_bic=True,csi_test="anderson",all_trees=True, use_dag=True)



#coronary_experiment_bic()

#susy_experiment(0.02)
coronary_experiment()
#synthetic_dag_binarydata_experiment(9, 0.2, 350)
#coronary_experiment()
#dermatology_experiment(grouped=True)
#micecortex_experiment()
#susy_experiment()


