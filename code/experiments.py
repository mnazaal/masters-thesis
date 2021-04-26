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
from models import CSTree

def synthetic_dag_binarydata_experiment(nodes, p_edge, n, use_dag=False):
    val_dict = binary_dict(nodes)
    dag = generate_dag(nodes, p_edge)
    dataset = synthetic_dag_binarydata(dag, n)

    cstree_object = CSTree(dataset)
    #ordering=None
    ordering=[i+1 for i in range(nodes)]
    cstree_object.visualize(ordering=ordering, use_dag=use_dag)
    
def coronary_experiment():
    # Load dataset
    dataset = coronary_data()
    
    # Create CSTree object
    cstree_object = CSTree(dataset)

    # Visualize CSTrees with fewest stages
    save_dir=None
    cstree_object.visualize(all_trees=False, save_dir=save_dir)


synthetic_dag_binarydata_experiment(10, 0.2, 1000)
#coronary_experiment()
