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


from utils.utils import contained, flatten, cpdag_to_dags, generate_vals, parents, dag_topo_sort,generate_state_space,shared_contexts,data_to_contexts,context_per_stage,get_size, binary_dict
from cstree import  cstree_pc, stages_to_csi_rels
from datasets import synthetic_dag_binarydata, bnlearn_data,coronary_data, dermatology_data, micecortex_data
from utils.utils import generate_dag, binary_dict, generate_state_space
from graphoid import graphoid_axioms
from mincontexts import minimal_context_dags, binary_minimal_contexts
from models import CSTree

def synthetic_dag_binarydata_experiment(nodes, p_edge, n, use_dag=False):
    dag = generate_dag(nodes, p_edge)
    dataset = synthetic_dag_binarydata(dag, n)

    cstree_object = CSTree(dataset)
    #ordering=None
    ordering=[i+1 for i in range(nodes)]
    cstree_object.visualize(ordering=ordering, use_dag=use_dag)


def dermatology_experiment(grouped=True):
    dataset = dermatology_data(grouped)

    cstree_object = CSTree(dataset)

    #cstree_object.learn(all_trees=False)

    cstree_object.visualize(use_dag=False)


def micecortex_experiment(num_features=10):
    dataset = micecortex_data(num_features)

    cstree_object = CSTree(dataset)

    cstree_object.visualize(all_trees=False, plot_limit=None, learn_limit=None)

    
    
    
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

dataset=coronary_data()
val_dict = binary_dict(6)

shape = tuple(len(list(val_dict.values())[i]) for i in range(len(val_dict)))
table = np.zeros(shape)

print(table.shape)


print(table[0,0,0,0,0,0])

print(dataset.shape)

print(dataset.tolist().count(list([0,0,0,0,0,0])))

for n in range(dataset.shape[0]):
    table[tuple(dataset[n,:])]+=1

test_array = np.random.rand(3,3,3)
print(test_array)
print(np.sum(test_array, axis=(2,1)))
