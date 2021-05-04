import math
import pytest

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from cstree import dag_to_cstree, stages_to_csi_rels, cstree_pc
from datasets import synthetic_dag_binarydata, coronary_data
from graphoid import decomposition, weak_union, weak_union, decomposition, graphoid_axioms
from algorithms import dag_to_ci_model
from mincontexts import binary_minimal_contexts,minimal_context_dags
from utils.utils import binary_dict, ternary_dict, mixed_dict, generate_dag, nodes_per_tree

# Arrange,  Act,  Assert
# pytst prefixtures




def dag_to_cstree_util(nodes, val_dict, ordering, dag, expected_stages):
    cstree, stages, colour_scheme = dag_to_cstree(val_dict, ordering, dag)
    if expected_stages==0 or expected_stages==nodes-1:
        assert len(stages)==expected_stages

    last_level_nodes = [n for n in cstree.nodes if nx.shortest_path_length(cstree,"Root",n)==len(ordering)-1]
    uncounted_last_level_nodes = len(last_level_nodes)*len(val_dict[ordering[-1]])
    assert len(cstree.nodes)+uncounted_last_level_nodes-1 == nodes_per_tree(val_dict, ordering)

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
    all_vars = []

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
        all_vars.append(a)
        assert a not in C
        assert a not in B
        for b in B:
            assert a>b
            assert b not in C
            

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

    

def test_weak_union_randdag():
    nodes = 10
    ordering = [i+1 for i in range(nodes)]
    val_dict = binary_dict(nodes)
    dag = generate_dag(nodes,0.3)
    # get csi rels  from dag to cstree model
    # generate own examples
    cstree, stages, colour_scheme = dag_to_cstree(val_dict, ordering=ordering, dag=dag)
    csi_rels = stages_to_csi_rels(stages, ordering)

    pairwise = True

   
    for csi_rel_fromtree in csi_rels:
        (A1,B1,S1,C1) = csi_rel_fromtree
        
        generated_rels = weak_union(csi_rel_fromtree, pairwise)
        #assert len(csi_rels)<len(generated_rels)

        if len(B1)==1:
            assert generated_rels==[]
        
        assert csi_rel_fromtree not in generated_rels
        
        for csi_rel_gen in generated_rels:
            (A2,B2,S2,C2) = csi_rel_gen
            # The second variable gets moved to conditioning
            # set so th below inequalities must hold
            assert len(B2)<len(B1)


def test_decomposition_randdag():
    nodes = 10
    ordering = [i+1 for i in range(nodes)]
    val_dict = binary_dict(nodes)
    dag = generate_dag(nodes,0.3)
    # get csi rels  from dag to cstree model
    # generate own examples
    cstree, stages, colour_scheme = dag_to_cstree(val_dict, ordering=ordering, dag=dag)
    csi_rels = stages_to_csi_rels(stages, ordering)

    pairwise = True

    for csi_rel_fromtree in csi_rels:
        (A1,B1,S1,C1) = csi_rel_fromtree
        
        generated_rels = decomposition(csi_rel_fromtree, pairwise)
        #assert len(csi_rels)<len(generated_rels)

        if len(B1)==1:
            assert generated_rels==[]
        
        assert csi_rel_fromtree not in generated_rels
        
        for csi_rel_gen in generated_rels:
            (A2,B2,S2,C2) = csi_rel_gen
            # The second variable gets moved to conditioning
            # set so th below inequalities must hold
            assert len(B2)<len(B1)

def  test_graphoid_pairwise():
    # TODO More tests for this
    # For each context we must have csi relations
    # satisfying certain properties
    nodes = 5
    ordering = [i+1 for i in range(nodes)]
    val_dict = binary_dict(nodes)
    dag = generate_dag(nodes,0.3)
    # get csi rels  from dag to cstree model
    # generate own examples
    cstree, stages, colour_scheme = dag_to_cstree(val_dict, ordering=ordering, dag=dag)
    csi_rels = stages_to_csi_rels(stages, ordering)

    closure = graphoid_axioms(csi_rels.copy(), val_dict)

    assert len(closure)>len(csi_rels)

def test_minimal_contexts_randdag():
    nodes    = 5
    ordering = [i+1 for i in range(nodes)]
    val_dict = binary_dict(nodes)
    dag = generate_dag(nodes,0.1)
    # get csi rels  from dag to cstree model
    # generate own examples
    cstree, stages, colour_scheme = dag_to_cstree(val_dict, ordering=ordering, dag=dag)
    csi_rels = stages_to_csi_rels(stages, ordering)

    closure = graphoid_axioms(csi_rels.copy(),val_dict)

   # print(len(csi_rels),len(closure))

    minimal_contexts = binary_minimal_contexts(closure, val_dict)
    assert list(minimal_contexts.keys())[0] == ()
"""
@pytest.mark.skip
def test_cstree_pc1():
    # TODO Look more into the following DAGs
    # They had a case where there was no CI relation for
    # the second and third variable in an ordering
    # But now, generating data from these DAGs is not
    # giving this problem
    # [(2, 4), (2, 6), (1, 2), (1, 4), (3, 7), (3, 5), (7, 5)]
    # [(7, 6), (7, 2), (2, 5), (4, 3), (4, 1)]  (Order [4, 1, 3, 7, 2, 6, 5])
    # [(2, 3), (7, 4), (1, 2), (1, 6), (5, 1)]
    # [(6, 2), (6, 3), (7, 1), (1, 5), (1, 4)]  (Order [7, 6, 1, 4, 2, 3, 5])
    # Problem: when we have disconnected components the
    # independencies are not encoded
    
    # Get some dataset
    # do the PC algorithm, get the DAG
    # Convert this DAG to CSTree

    # Get the CSTree from the CSTree PC algorithm
    # Determine whether
    # 1. Stages did not increase
    # 2. Minimal contexts >0

    # TODO nodes 5 sometimes gave  some problems
    nodes = 7
    val_dict = binary_dict(nodes)
    #dag = generate_dag(nodes,0.2)
    #nodes in DAG ust equal to th edges less
    dag = nx.DiGraph()
    dag.add_edges_from( [(2, 3), (7, 4), (1, 2), (1, 6), (5, 1)])
    #dag.add_nodes_from([i+1 for i in range(nodes)])

    data = synthetic_dag_binarydata(dag, 500)
    print("starting pc")
    cstree_pc(data, val_dict)

def test_cstree_pc2():
    val_dict = binary_dict(7)
    dag = nx.DiGraph()
    dag.add_edges_from( [(2, 4), (2, 6), (1, 2), (1, 4), (3, 7), (3, 5), (7, 5)])
    data = synthetic_dag_binarydata(dag, 200)
    cstree_pc(data, val_dict)

def test_cstree_pc3():
    val_dict = binary_dict(7)
    dag = nx.DiGraph()
    dag.add_edges_from([(6, 2), (6, 3), (7, 1), (1, 5), (1, 4)])
    data = synthetic_dag_binarydata(dag, 1000)
    cstree_pc(data, val_dict)

def test_cstree_pc4():
    val_dict = binary_dict(7)
    dag = nx.DiGraph()
    dag.add_edges_from( [(7, 6), (7, 2), (2, 5), (4, 3), (4, 1)])
    data = synthetic_dag_binarydata(dag, 500)
    cstree_pc(data, val_dict)
    
def test_cstree_pc5():
    val_dict = binary_dict(7)
    dag = nx.DiGraph()
    dag.add_nodes_from( [i+1 for i in range(7)])
    data = synthetic_dag_binarydata(dag, 500)
    cstree_pc(data, val_dict)

def test_cstree_pc6():
    val_dict = binary_dict(7)
    dag  = generate_dag(7,1)
    data = synthetic_dag_binarydata(dag, 500)
    cstree_pc(data, val_dict)
    """

def test_dag_v_nodag():
    val_dict = binary_dict(6)
    data = coronary_data()

def test_binary_minimal_contexts1():
    val_dict = binary_dict(6)
    csi_rels = [({2},{1},set(),[(6,1),(3,1)]), ({2},{1},set(), [(6,1),(3,0)])]
    assert list(binary_minimal_contexts(csi_rels, val_dict).keys())[0] == ((6,1),)

def test_binary_minimal_contexts2():
    val_dict = binary_dict(4)
    csi_rels = [({4},{2},set(),[(1,0),(3,0)]), ({4},{2},set(),[(1,0),(3,1)]),({4},{2},set(),[(1,1),(3,0)]),({3},{1},set(),[(2,0)])]
    min_context_dict = binary_minimal_contexts(csi_rels,val_dict)
    min_contexts = set(list(min_context_dict.keys()))    
    assert min_contexts == {((1,0),),((2,0),),((3,0),)}

def test_ternary_minimal_contexts1():
    val_dict = ternary_dict(4)
    csi_rels = [({4},{1}, set(), [(1,0)]), ({4},{1},set(),[(1,1)])]
    # binary here refers to binary search
    min_context_dict = binary_minimal_contexts(csi_rels, val_dict)
    min_contexts = set(list(min_context_dict.keys()))
    assert min_contexts == {((1,0),), ((1,1),)}

def test_ternary_minimal_contexts2():
    val_dict = ternary_dict(4)
    csi_rels = [({4},{1}, set(), [(1,0)]), ({4},{1},set(),[(1,1)]), ({4},{1},set(),[(1,2)])]
    # binary here refers to binary search
    min_context_dict = binary_minimal_contexts(csi_rels, val_dict)
    min_contexts = set(list(min_context_dict.keys()))
    assert min_contexts == {()}

    
def test_weak_union():
    pass
#if __name__ == "__main__":
    
