# Python imports
import sys
from functools import reduce
from itertools import product

# Third-party library imports
import networkx as nx
import numpy as np

# Project imports
from legacy import Variable, Context, Distribution, CiRelation



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


def nodes_per_tree(val_dict, ordering):
    assert len(val_dict)==len(ordering)

    # nodes per each level
    nodes=[]
    nodes.append(len(val_dict[ordering[0]]))
    
    for i in range(1,len(ordering)):
        nodes.append(nodes[i-1]*len(val_dict[ordering[i]]))
        
    return sum(nodes)


def contained(p, rels, s):
    # checking whether there is any s' in the conditioning
    # subsets already d-separating the pair p such that 
    # s' is contained in the new subset s
    # because in this case we already know they are 
    # d-separated
    
    # take the relevant ci relations
    relevant = list(filter(lambda r: r[0] == p, rels))
    
    # get the ones where an existing conditioning set is a subset of s
    r = list(map(lambda r: s.issubset(r[-1]), relevant))
    
    return not reduce(lambda a,b:a and b, r, True)



def flatten(T,a):
    if not isinstance(T,tuple): return (T,)
    else:
        for i in T:
            if isinstance(i, (list,tuple)):
                for j in flatten(i,a):
                    yield j
            else:
                yield i


# reduce for and
and_ = lambda l:functools.reduce(lambda x,y:x and y, l, False)

# check if two edges are undirected versions of each other
edge_opposite = lambda e1,e2:  e1[0]==e2[1] and e1[1]==e2[0]               

def v_structure(e1, e2, g):
    # Return true if two edges form a v-structure
    # check if they are head to head, AND uncoupled
    heads_meet = e1[1]==e2[1]
    tails_dont = e1[0]!=e2[0]
    all_edges  = list(g.edges)
    no_other_tail_edge = ((e1[0], e2[0]) not in all_edges) or ((e2[0], e1[0]) not in all_edges) 
    return heads_meet and tails_dont and no_other_tail_edge


def coming_in(edge, g):
    es = list(g.edges)
    return [e for e in es if e[1]==edge[1]]

def v_structure_graph(g):
    es = list(g.edges)
    all_pairs = list(itertools.combinations(es,2))
    return and_(list(map( lambda x: v_structure(x[0],x[1], g), all_pairs)))

def undirected(es):
    es=list(es)
    n = len(es)
    u_es = []
    for i in range(n):
        for j in range(i,n):
            if es[i][0] == es[j][1] and es[i][1] == es[j][0]:
                u_es.append(es[i])
    return u_es

def directed(es):
    u_es = undirected(es)
    all_u_es = u_es+[(y,x) for (x,y) in u_es]
    return [d_e for d_e in es if d_e not in all_u_es]


def cpdag_to_dags(g):
    
    undirected_edges = undirected(g.edges)
    directed_edges   = directed(g.edges)    

    if undirected_edges == []:
        try:
            if nx.find_cycle(g):
                raise ValueError("Found cycle, CPDAG might be incorrect")
        except:
            yield g
    else:
        
        u = undirected_edges.pop()
        # pick just the first undirected edge, then recursively move on

        # Test first for the orientation (u,v)
        u1 = (u[0],u[1])
        # Get all the edges that come into v in g
        
        # TODO Generator here
        connected_u1 = coming_in(u1,g)

        # Filter the edges which were originally directed in the CPDAG
        
        # TODO Generator here
        c_and_d1 = [i for i in connected_u1 if i in directed_edges]

        # If the above list is empty, no possible v structure can be formed,
        # so set (u,v) as a directed by (by removing (v,u)) and recurse
        if not c_and_d1:
            new_g3 = g.copy()
            new_g3.remove_edge(u[1],u[0])
            yield from cpdag_to_dags(new_g3)
        else:
            # If there are edges of the form (_,v)
            for e in c_and_d1:
                # If they do not form a v structure in g, we are good
                if not v_structure(u1,e,g):
                    new_g1 = g.copy()
                    # Set (u,v) as a directed edge (by removing (v,u)) and recurse
                    new_g1.remove_edge(u[1],u[0])
                    yield from cpdag_to_dags(new_g1)

        # Then for the orientation (v,u)
        u2 = (u[1],u[0])
        connected_u2 = coming_in(u2,g)            
        c_and_d2 = [i for i in connected_u2 if i in directed_edges]
        if not c_and_d2:
            new_g4 = g.copy()
            new_g4.remove_edge(u[0],u[1])
            yield from cpdag_to_dags(new_g4)
        else:
            for e in c_and_d2:
                if not v_structure(u2,e,g):
                    new_g2 = g.copy()
                    new_g2.remove_edge(u[0],u[1])
                    yield from cpdag_to_dags(new_g2)
                    
                    
def parents(g,node):
    return list(g.predecessors(node))

def generate_vals(T, state_space_dict):
    # generate all possible values for the variables in subset T 
    # list of lists containing the values 
    s_vals = [state_space_dict[t] for t in T]
    all_vals_prod = list(product(*s_vals)) # all values without variables in front 
    f = lambda tup: [(T[i], tup[i]) for i in range(len(T))]
    return list(map(f,all_vals_prod))


def generate_state_space(data):
    # Works under the assumption that the data has all values for each variable occuring atleast once
    return {i+1:list(np.unique(data[:,i])) for i in range(data.shape[1])}

dag_topo_sort = lambda dag: list(nx.topological_sort(dag))


def shared_contexts(cs1,cs2):
    return [c for c in cs1 if (c in cs1 and c in cs2)]



def context_per_stage(ls):
    # Takes nodes in a stage and returns the common context they represent
    sc = shared_contexts(ls[0],ls[1])
    for l in ls[2:]:
        sc = shared_contexts(sc,l)
    return sc

def data_to_contexts(data : np.ndarray, 
                     cs   : list[tuple[int,int]], 
                     var  : int) -> np.ndarray:
    # returns the data where the contexts are of the relevant form
    # confirm if same contexts havent been put
    # TODO consider moving to dask later on
    
    # NOTE
    # When accessing columns of data, we must reduce 1 since data matrix indexing starts with 0
    # and the var indexing starts from 1
    
    p = data.shape[1]
    #c=cs[0]
    for c in cs:
        if var == c[0]:
            raise ValueError("The variable you want is being fixed in the context")
        data = data[np.where(data[:,c[0]-1]==c[1])]
    return data[:,var-1] #-1 because indices are labelled from 0


def context_is_contained(c, cs):
    # Given a context c, and a list of contexts cs,
    # return the context x in cs if c is a subcontext of x
    contained=None
    for context in cs:
        
        if set(c).issubset(context):
            contained=context
            break
    return contained


def get_size(obj, seen=None):
    """Recursively finds size of objects
    Code from stackoverflow"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
