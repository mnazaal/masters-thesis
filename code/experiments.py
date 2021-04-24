# Generating synthetic data from a random DAG
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pgmpy
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, MmhcEstimator
from pgmpy.base import DAG
from pgmpy.independencies import Independencies


from cstree import cstree_pc
import time
import itertools

import numpy as np
import networkx as nx
from gsq.gsq_testdata import bin_data, dis_data
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import bnlearn
from pgmpy.factors.discrete import TabularCPD

from utils import contained, flatten, cpdag_to_dags, generate_vals, parents, dag_topo_sort,generate_state_space,shared_contexts,data_to_contexts,context_per_stage,get_size
from cstree import dag_to_cstree, cstree_pc, cstree_pc1
from algorithms import dag_to_ci_model
from mincontexts import minimal_contexts1, minimal_context_dags1


def synthetic_dag():
    p   = 6
    dag = nx.gnp_random_graph(p,0.9,directed=True)
    dag = nx.DiGraph([(u,v) for (u,v) in dag.edges() if u<v])
    dag = nx.relabel_nodes(dag, lambda x: x +1)
    for i in range(1,p+1):
        if i not in dag.nodes:
            dag.add_node(i)

    #dag = nx.DiGraph()
    #dag.add_edges_from([(i+1,i+2) for i in range(4)])
    #dag.add_nodes_from([i for i in range(5,10)])

    #dag = nx.DiGraph()
    #dag.add_edges_from([(1,2),(1,3),(2,4),(3,4),(4,5)])

    syntheticdata_dag=dag
    order = dag_topo_sort(syntheticdata_dag)
    value_dict = {var:[0,1] for var in order}
    print(order)
    options = {
        'node_color': 'white',
        'node_size': 1000,
    }
    plt.figure(figsize=(7,7))
    nx.draw_networkx(syntheticdata_dag,pos=nx.drawing.layout.shell_layout(syntheticdata_dag), **options)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000") 
    plt.savefig("figs/dag_syntheticdata"+str(p)+".pdf")
    syntheticdata_dag = nx.relabel_nodes(syntheticdata_dag, lambda x: str(int(x)))

    # ideas from https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/2.%20Bayesian%20Networks.ipynb
    no_parents = [n for n in list(syntheticdata_dag.nodes) if parents(syntheticdata_dag,n)==[]]
    cpds = []
    for var in no_parents:
        pr = np.random.rand()
        cpds.append(TabularCPD(variable=var, variable_card=2, values = [[pr],[1-pr]]))

    has_parents = [n for n in list(syntheticdata_dag.nodes) if n not in no_parents]


    for var in has_parents:

        parents_var  = parents(syntheticdata_dag,var)

        rows = [i+1 for i in range(2**len(parents_var))]
        p_table = np.zeros((2,rows[-1]))
        """
        print(rows[-1])
        p_table_row1 = np.zeros((1,rows[-1]))
        context_row=1
        for r in rows:
            if r==context_row:
                p_table_row1[0,r-1]=1
            else:
                p = float(r)/(rows[-1]+1)
                p_table_row1[0,r-1]=p
        print(p_table_row1)
        p_table_row1 = np.random.rand(1,2**len(parents_var))

        """

        row_cutoff=2**(len(parents_var)-1)
        #row_cutoff = np.random.randint(1, 2**(len(parents_var)))

        p_table_row1half1 = 0.9*np.ones((row_cutoff))
        p_table_row1half2 = 0.1*np.ones((2**(len(parents_var))-row_cutoff))
        p_table_row1 = np.concatenate((p_table_row1half1,p_table_row1half2)).reshape(2**len(parents_var))
        #if 1 in parents_var:
        #    p_table_row1[0] = 0
        p_table_row2 = np.ones((1,2**len(parents_var)))-p_table_row1
        p_table[0,:]  = p_table_row1
        p_table[1,:]  = p_table_row2
        cpds.append(TabularCPD(variable=var, variable_card=2, 
                       values=p_table.tolist(),
                      evidence=parents_var,
                      evidence_card=[2]*len(parents_var)))


    DAG = bnlearn.make_DAG(list(syntheticdata_dag.edges), CPD=cpds)
    df = bnlearn.sampling(DAG, n=1000)

    print("Variables are ",df.columns.values)
    bnlearn_data = df.values[1:,:]
    data = np.zeros((bnlearn_data.shape[0], bnlearn_data.shape[1]+3))
    data[:,:bnlearn_data.shape[1]]=bnlearn_data
    data[:,bnlearn_data.shape[1]:]=np.random.randint(0,2,size=(bnlearn_data.shape[0], +3))
    #bnlearn_data = bnlearn_data.astype(np.int)
    data = data.astype(np.int)
    #cstree_pc(bnlearn_data,  draw=True, test="epps")
    print(bnlearn_data)
    
    
def synthetic_cpt():
    x_1 =TabularCPD(variable='1', variable_card=2, values = [[0.5],[1-0.5]])
    px_31 = 0.99
    px_32 = 0.1  # probability of x_3 being 1 in the second context
    x_3 = TabularCPD(variable='3', variable_card=2, values=[[px_31,     px_32], 
                                                          [1-px_31, 1-px_32]], evidence=['1'], evidence_card=[2])
    px_21 = 0.9
    px_22 = 0.1
    x_2 = TabularCPD(variable='2', variable_card=2, values=[[px_21, px_22], 
                                                          [1-px_21, 1-px_22]], evidence=['1'], evidence_card=[2])

    px_41 = 0.01
    px_42 = 0.2
    px_43 = 0.5
    px_44 = 0.99
    x_4 = TabularCPD(variable='4', variable_card=2, values=[[px_41,     px_42,   px_43, px_44], 
                                                          [1-px_41, 1-px_42,1-px_43,1-px_44]], evidence=['2','3'], evidence_card=[2,2])

    px_51 = 0.9
    px_52 = 0.01
    x_5 = TabularCPD(variable='5', variable_card=2, values=[[px_51, px_52], 
                                                          [1-px_51, 1-px_52]], evidence=['4'], evidence_card=[2])


    DAG = bnlearn.make_DAG([('1','2'),('1','3'),('2','4'),('3','4'),('4','5')], CPD=[x_1,x_2,x_3,x_4,x_5])
    df = bnlearn.sampling(DAG, n=100000)
    bnlearn_data = df.values[1:,:]
    bnlearn_data = bnlearn_data.astype(int)
    
def adjacency_method():
    p = 10
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i>j:
                A[i,j] = np.random.binomial(1, 0.5)
                if A[i,j] == 1:
                    A[i,j] = np.random.uniform(0.1,1)

    n = 1000

    sign = np.vectorize(lambda x: 1 if x>0 else 0)

    def sampler(A,p):
        val = np.zeros(p)
        val[0] = np.random.normal(0,1)
        for i in range(1,p):
            val[i] = sum([A[i,k-1]*val[k-1] for k in range(i)])
        return sign(val)

    n_samples = lambda n: np.array([sampler(A,p) for i in range(n)])
    
    
def coronary():
    coronary         = pd.read_csv("coronary.csv")
    coronary.columns = ["id:0", "S:1", "MW:2", "PW:3", "P:4", "L:5", "F:6"]
    print(coronary)
    coronary_dict = {"yes":1, "no":0, "<140":0, ">140":1, "<3":0, ">3":1, "neg":0, "pos":1}
    coronary=coronary.replace(coronary_dict)
    coronary = coronary.values[1:,1:]
    coronary_np = coronary.astype(int)
    #cstree_pc(dataset_all,draw=True,test="epps")
    coronary = pd.DataFrame(coronary_np, columns=[i for i in range(1,coronary_np.shape[1]+1)])
    coronary_model_pgmpy= PC(coronary)
    coronary_dag_pgmpy=coronary_model_pgmpy.estimate(return_type="cpdag")
    coronary_dag_nx=nx.DiGraph()
    es= list(coronary_dag_pgmpy.edges())
    coronary_dag_nx.add_edges_from(es)
    coronary_dag_nx.add_nodes_from([i for i in range(1, coronary_np.shape[1]+1)])
    
def bnlearn_datasets():
    loadfile1 = 'sprinkler'
    loadfile2 = 'alarm'
    loadfile3 = 'andes'
    loadfile4 = 'asia'
    loadfile5 = 'pathfinder'
    loadfile6 = 'sachs'
    loadfile7 = 'miserables'

    DAG = bnlearn.import_DAG(loadfile4)

    df = bnlearn.sampling(DAG, n=1000)
    print("Variables are ",df.columns.values)
    bnlearn_data = df.values[1:,:]
    bnlearn_data = bnlearn_data.astype(np.int)
    #cstree_pc(bnlearn_data, draw=False)
