# Python imports
import itertools, functools
from itertools import product

# Third party library imports
import jax
import networkx as nx

# Project imports
from legacy import Variable, Context, Distribution, CiRelation
from utils import contained, flatten

# TODO Write wrappers which uses bnlearn, causaldag, pc packages
# rather than calling them directly in between main code

def dag_to_ci_model(d: nx.DiGraph) -> list[CiRelation]:
    # TODO Cast into proper CiRelation type later
    # Convert DAG to CI relations
    ci_relations = []
    p = len(d.nodes)
    nodes = list(range(1,p+1))
    
    # List of all pairs of nodes
    all_pairs = [(i) for i in itertools.combinations(nodes, 2)]
    
    subset_size = 0
    while subset_size <= p:
        # for each possible subset sizes 
        # TODO Maybe write a JIT function here
        
        # Generate all subsets of that size
        subsets = [set(i) for i in itertools.combinations(nodes, subset_size)]
        if subset_size == 0:
            subsets = [set()]
        # TODO vmap over all possible subset, pair combinations?
        for s in subsets:
            for pair in all_pairs:
                if set(pair).isdisjoint(s):
                    if not contained(pair, ci_relations, s): # if s is not a superset of any of the known d-separating sets of this pair
                        if nx.d_separated(d, {pair[0]}, {pair[1]}, s):
                            ci_relations.append((set(pair), s))               
            
        subset_size +=1
    return ci_relations

