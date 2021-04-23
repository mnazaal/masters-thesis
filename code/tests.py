import math
import pytest

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from cstree import dag_to_cstree, stages_to_csi_rels, decomposition, weak_union, cstree_pc1
from algorithms import dag_to_ci_model
from mincontexts import binary_minimal_contexts1,binary_minimal_contexts1,minimal_context_dags1

# Arrange,  Act,  Assert
# pytst prefixtures

# State space dictionaries
# binary, ternary, random ints, one dict with 1 value
binary_dict  = lambda p: {i+1:[0,1] for i in range(p)}
ternary_dict = lambda p: {i+1:[0,1,2] for i in range(p)}
mixed_dict   = lambda p: {i+1:[0,1] if i < p//2 else [0,1,2] for i in range(p)}
# CSI relations

def generate_dag(nodes, p_edge):
    rand_graph = nx.gnp_random_graph(nodes,p_edge,directed=True)
    dag        = nx.DiGraph()
    dag.add_edges_from([(u,v) for (u,v) in rand_graph.edges if u<v])
    dag.add_nodes_from([i for i in range(nodes)])
    dag = nx.relabel_nodes(dag, lambda x: x+1)
    return dag

def nodes_per_tree(val_dict):
    dict_as_items = list(val_dict.items())
    if len(val_dict)==2:
        return len(dict_as_items[0][-1])
    else:
        return len(dict_as_items[0][-1]) +  len(dict_as_items[0][-1])* nodes_per_tree({k:v for k,v in dict_as_items[1:]})


def dag_to_cstree_util(nodes, val_dict, ordering, dag, expected_stages):
    cstree, stages, colour_scheme = dag_to_cstree(val_dict, ordering, dag)
    if expected_stages==0 or expected_stages==nodes-1:
        assert len(stages)==expected_stages
    assert len(cstree.nodes) == nodes_per_tree(val_dict)+1

    coloured_nodes = []
    for c,ns in stages.items():
        coloured_nodes +=ns
        level = len(ns[0])
        for node in ns:
            assert len(node)==level

    assert len(colour_scheme)==len(coloured_nodes)

    
def test_dag_to_cstree_emptydag():
    nodes=5
    dag_to_cstree_util(nodes, binary_dict(nodes), [i+1 for i in range(nodes)], generate_dag(nodes,0),nodes-1)
    nodes=3
    dag_to_cstree_util(nodes, ternary_dict(nodes), [i+1 for i in range(nodes)], generate_dag(nodes,0),nodes-1)
    dag_to_cstree_util(nodes, mixed_dict(nodes), [i+1 for i in range(nodes)], generate_dag(nodes,0),nodes-1)


def test_dag_to_cstree_fulldag():
    nodes  = 5
    dag_to_cstree_util(nodes, binary_dict(nodes), [i+1 for i in range(nodes)], generate_dag(nodes,1),0)
    nodes  = 3
    dag_to_cstree_util(nodes, ternary_dict(nodes), [i+1 for i in range(nodes)], generate_dag(nodes,1),0)
    dag_to_cstree_util(nodes, mixed_dict(nodes), [i+1 for i in range(nodes)], generate_dag(nodes,1),0)


def test_dag_to_cstree_randdag():
    nodes  = 5
    dag_to_cstree_util(nodes, binary_dict(nodes), [i+1 for i in range(nodes)], generate_dag(nodes,0.5),-1)
    nodes  = 3
    dag_to_cstree_util(nodes, ternary_dict(nodes), [i+1 for i in range(nodes)], generate_dag(nodes,0.5),-1)
    dag_to_cstree_util(nodes, mixed_dict(nodes), [i+1 for i in range(nodes)], generate_dag(nodes,0.5),-1)


def stages_to_csi_rels_util(nodes, val_dict, ordering, dag):
    cstree, stages, colour_scheme = dag_to_cstree(val_dict, ordering=ordering, dag=dag)
    csi_rels = stages_to_csi_rels(stages, ordering)
    assert len(csi_rels)==len(stages)

    # Testing each CSI relation is of the form
    # X_k _||_ X_[k-1]\C | X_C=x_c
    for csi_rel in csi_rels:
        # csi_rel =  (A,B,S,C) for X_A_||_X_B|X_S,X_C=x_C
        A = csi_rel[0]
        B = list(csi_rel[1])
        S = csi_rel[2]
        C = [var for (var,val) in csi_rel[-1]]

        # Their intersection must be non zero
        assert set.intersection(set(A),set(B),set(S),set(C)) == set()
        
        # Conditions of A, |A|=1
        assert len(A)==1

        # Each element in B must come before A
        # and not be in C
        a = A.pop()
        for b in B:
            assert a>b
            assert b not in C
            assert a not in C

        # S must be empty
        assert len(S)  == 0

        # Each variable in context must come before a
        for c in C:
            assert a>c

def test_stages_to_csi_rels_emptydag():
    nodes=5
    stages_to_csi_rels_util(nodes, binary_dict(nodes),  [i+1 for i in range(nodes)], generate_dag(nodes,0))
    nodes=3
    stages_to_csi_rels_util(nodes, ternary_dict(nodes),   [i+1 for i in range(nodes)], generate_dag(nodes,0))
    stages_to_csi_rels_util(nodes, mixed_dict(nodes),   [i+1 for i in range(nodes)], generate_dag(nodes,0))


def test_stages_to_csi_rels_fulldag():
    nodes=5
    stages_to_csi_rels_util(nodes, binary_dict(nodes),  [i+1 for i in range(nodes)], generate_dag(nodes,1))
    nodes=3
    stages_to_csi_rels_util(nodes, ternary_dict(nodes),   [i+1 for i in range(nodes)], generate_dag(nodes,1))
    stages_to_csi_rels_util(nodes, mixed_dict(nodes),   [i+1 for i in range(nodes)], generate_dag(nodes,1))

    
def test_stages_to_csi_rels_randdag():
    nodes=5
    stages_to_csi_rels_util(nodes, binary_dict(nodes),  [i+1 for i in range(nodes)], generate_dag(nodes,0.2))
    nodes=3
    stages_to_csi_rels_util(nodes, ternary_dict(nodes),   [i+1 for i in range(nodes)], generate_dag(nodes,0.2))
    stages_to_csi_rels_util(nodes, mixed_dict(nodes),   [i+1 for i in range(nodes)], generate_dag(nodes,0.2))



def colour_cstree_util(tree, ordering, data, stages, colour_scheme, no_dag):
    learnt_cstree, learnt_stages, learnt_colour_scheme = colour_cstree(tree, ordering, data, stages,colour_scheme,no_dag=no_dag)
    assert len(learnt_stages)<len(stages)
    assert len(learnt_colour_scheme)<len(colour_scheme)
    assert len(learnt_stages)==len(learnt_colour_scheme)

    # Each stage must have nodes in the same level
    for c,ns in learnt_stages.items():
        level = len(ns[0])
        for node in ns:
            assert len(node)==level

    # if gave a non-singleton stage from any level,
    # we should have atmost that much non-singleton stages at each level
    for level in range(1,len(ordering)+1):
        learnt_stages_level = {c:ns for c,ns in learnt_stages.items() if len(ns[0])==level}
        stages_level        = {c:ns for c,ns in stages.items() if len(ns[0])==level}
        assert learnt_stages_level <= stages_level

def test_colour_cstree_randdata_nodag():
    pass

def test_colour_cstree_randdata_wdag():
    pass

def test_colour_cstree_randdata():
    pass

def test_colour_cstree():
    # TODO Generate datasets for this
    # Number of stages can only decrease
    # Colouring empty trees
    # Colouring fully connected trees
    # Test number of nodes being equal


    pass

    

def test_decomposition_wdag():
    nodes = 10
    ordering = [i+1 for i in range(nodes)]
    val_dict = binary_dict(nodes)
    dag = generate_dag(nodes,0.3)
    # get csi rels  from dag to cstree model
    # generate own examples
    cstree, stages, colour_scheme = dag_to_cstree(val_dict, ordering=ordering, dag=dag)
    csi_rels = stages_to_csi_rels(stages, ordering)

    generated_rels = decomposition(csi_rels)

    assert len(csi_rels)<=len(generated_rels)

    
    for csi_rel1 in csi_rels:
        B1 = csi_rel1[1]
        S1 = csi_rel1[2]
        assert csi_rel1 not in generated_rels
        for csi_rel2 in generated_rels:
            B2 = csi_rel2[1]
            S2 = csi_rel2[2]
            assert len(B2)<len(B1)
            assert len(S2)>len(S1)
    
    
    

    # make sure we get atleast that
    
    pass


def test_weak_union():
    pass
#if __name__ == "__main__":
    
