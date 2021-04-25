import numpy as np
import networkx as nx
from pgmpy.factors.discrete import TabularCPD
import bnlearn

from utils import dag_topo_sort, parents

def synthetic_dag_binarydata(dag_received, n):
    p = len(dag_received.nodes)
    # Getting nodes with no edges to sample later on
    separate_nodes = []
    for node in dag_received.nodes:
        if list(dag_received.in_edges(node)) ==  list(dag_received.out_edges(node)):
            separate_nodes.append(node)

    # Making the dag only with nodes having edges
    dag = nx.DiGraph()
    dag_edges = [e for e in dag_received.edges]
    dag.add_edges_from(dag_edges)
    
    dag = nx.relabel_nodes(dag, lambda x: str(int(x)))
    
    ordering = dag_topo_sort(dag)

    vars_w_no_parents = [n for n in list(dag.nodes) if parents(dag,n)==[]]

    cpds = []


    for var in vars_w_no_parents:
        pr = np.random.rand()
        cpds.append(TabularCPD(variable=var,
                               variable_card=2,
                               values = [[pr],
                                         [1-pr]]))

    vars_w_parents = [i for i in ordering if i not in vars_w_no_parents]

    for var in vars_w_parents:
        parents_var = parents(dag,var)

        num_rows = 2**(len(parents_var))

        p_table  = np.zeros((2, num_rows))

        for row in range(num_rows):

            low_p  = np.random.uniform(0.01, 0.2)
            high_p = np.random.uniform(0.8,0.99)

            coin_flip = np.random.rand()
            if coin_flip<0.5:
                pr = low_p
            else:
                pr = high_p

            p_table[0,row] = pr
            p_table[1,row] = 1-pr

        cpds.append(TabularCPD(variable=var,
                               variable_card=2,
                               values = p_table.tolist(),
                               evidence = parents_var,
                               evidence_card = [2]*len(parents_var)))

        
    df  = bnlearn.sampling(bnlearn.make_DAG(list(dag.edges), CPD=cpds), n)

        
    dataset = np.zeros((n,p))
        
        # Adding samples from nodes with no edges
    for i in range(p):
        var = i+1
        if var in separate_nodes:
            pr      = np.random.rand()
            samples = np.random.randint(0,2,size=(n,))
        else:
            samples = df[str(var)]
        dataset[:,i] = samples

    return dataset.astype(np.int)
        

        

        
