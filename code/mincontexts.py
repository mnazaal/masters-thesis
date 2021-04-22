from itertools import combinations
import time

import networkx as nx

from utils import generate_vals, context_is_contained
            
union = lambda sets : set.union(*sets)

def recursive_search(T_sizes, C, contexts_with_p, val_dict):
    if len(T_sizes) == 1:
        return T_sizes[0]
    
    else:
        mid = int(len(T_sizes)/2)
        candidate_size = T_sizes[mid]
        possible_Ts    = [set(i) for i in combinations(C, candidate_size)]
        for T in possible_Ts:
            vals_T_takes = generate_vals(list(T), val_dict)
            
            x_T_count=0
            for i, x_T in enumerate(vals_T_takes):
                contained  = context_is_contained(x_T, contexts_with_p)
                if contained:
                    x_T_count += 1
                
                if not contained:
                    return recursive_search(T_sizes[:mid], C, contexts_with_p, val_dict)
                if x_T_count==len(vals_T_takes):
                    return recursive_search(T_sizes[mid:], C, contexts_with_p, val_dict)
        

    
def binary_minimal_contexts1(csi_rels, val_dict, pairwise=True):
    minimal_cs      = set()
    minimal_cs_dict = {}
    
    
    
    # We get all the unique pairs involved in all the csi relations with non-empty contexts
    # TODO Think about making pairs a set instead of tuple
    
    if pairwise:
        mcs   = set()
        pairs = [union([A,B]) for (A,B,S,C) in csi_rels if len(A)==1 and len(B)==1]
        
        for p in pairs:
            csi_rels_with_p = [c for c in csi_rels if union([c[0],c[1]])==p]
            
            contexts_with_p = [c[-1] for c in csi_rels_with_p]
            done_for_rel = False
            for csi_rel in csi_rels_with_p:
                # Context variables of current CSI relation
                C = [var for (var, val) in csi_rel[-1]]
                
                # Possible sizes for T
                possible_T_sizes = [i for i in range(len(C)+1)] #includes empty set and fullset
                                
                # True size of T
                T_size = recursive_search(possible_T_sizes, C, contexts_with_p.copy(), val_dict)
                
                # Find the T
                if T_size == 0:
                    T = []
                elif T_size == len(C):
                    T = C.copy()
                else:
                    possible_Ts = [set(i) for i in combinations(C, T_size)]
                    for possible_T in possible_Ts:
                        vals_T_takes = generate_vals(list(T), val_dict)
                        
                        x_T_count = 0
                        for i, x_T in enumerate(vals_T_takes):
                            contained  = context_is_contained(x_T, contexts_with_p.copy())
                            if contained:
                                x_T_count +=1
                            if not contained:
                                break
                            if x_T_count==len(vals_T_takes):
                                T = possible_T
                                break
                
                context_of_rel  = csi_rel[-1]
                minimal_context = tuple(tuple(c) for c in context_of_rel if c[0] not in T)
                A = list(p)[0]
                B = list(p)[1]
                ci_rel_remaining = ({A},{B}, set(T).difference(set(C)))
                if minimal_context in minimal_cs_dict.keys():
                    if ci_rel_remaining not in minimal_cs_dict[minimal_context]:
                        minimal_cs_dict[minimal_context].append(ci_rel_remaining)
                else:
                    minimal_cs_dict[minimal_context] = [ci_rel_remaining]                    

            
    return minimal_cs_dict

def minimal_context_dags1(order, csi_rels, val_dict, extra_rels=None):
    mc_time = time.time()
    minimal_contexts=binary_minimal_contexts1(csi_rels, val_dict)
    minimal_context_dags = []
    
    for minimal_context, ci_rels in minimal_contexts.items():
        minimal_context_dag = nx.DiGraph()
        C = [var for (var,val) in minimal_context]
        remaining_vars = [var for var in order if var not in C]
        nodes = len(remaining_vars)
        for i in range(nodes):
            for j in range(i+1, nodes):
                pi_i = remaining_vars[i]
                pi_j = remaining_vars[j]
                
                minimal_context_dag.add_edge(pi_i,pi_j)
                
                conditioning_set = set(order[k] for k in range(j-1)).difference(set([pi_i])).difference(set(C))
                
                for ci_rel in ci_rels:
                    A = ci_rel[0]
                    B = ci_rel[1]
                    C = ci_rel[-1]
                    if A.union(B) == {pi_i,pi_j} and C.issubset(conditioning_set):
                        if minimal_context_dag.has_edge(pi_i,pi_j):
                            minimal_context_dag.remove_edge(pi_i,pi_j)
        minimal_context_dags.append((minimal_context, minimal_context_dag))
        
    return minimal_context_dags
        
    
    
    
    

    print(" minimal contexts took ", time.time()-mc_time)
    all_mc_graphs = []
    for mc, ci_rels in mcs.items():
        #print("CI rels before", ci_rels)
        if extra_rels:
            for rel in extra_rels:
                if rel[3]==list(mc):
                    a = next(iter(rel[0]))
                    b = next(iter(rel[1]))
                    s = rel[2]
                    ci_rels.append(({a,b},s))
        #print("ci rels after ", ci_rels)
        #print("MC and CI rels", mc, ci_rels)
        mc_graph       = nx.DiGraph()
        removed_edges  = 0
        added_edges    = 0
        mc_vars        = [i for (i,j) in mc]
        remaining_vars = [o for o in order if o not in mc_vars]
        mc_graph_nodes = len(remaining_vars)
        for i in range(mc_graph_nodes):
            for j in range(i+1, mc_graph_nodes):
                # These are indices for the list of ordering
                pi_i = remaining_vars[i]
                pi_j = remaining_vars[j]
                
                mc_graph.add_node(pi_i)
                mc_graph.add_node(pi_j)
                
                mc_graph.add_edge(pi_i, pi_j)
                added_edges+=1
                conditioning_set = set(order[k] for k in range(j-1)).difference(set([pi_i])).difference(set(mc_vars))
                
        
        
                #print("ij con are",pi_i,pi_j, conditioning_set, ci_rels)
                
                # MC DAGs Case a
                
                for ci_rel in ci_rels:
                    ci_rel_pair    = ci_rel[0]
                    ci_rel_condset = ci_rel[1]
                    if ci_rel_pair == set([pi_i, pi_j]) and ci_rel_condset.issubset(conditioning_set):
                        if mc_graph.has_edge(pi_i,pi_j):
                            mc_graph.remove_edge(pi_i, pi_j)
                            removed_edges+=1
                        else:
                            pass
                            #print("hit an alrady removed iedge")
                        
                
                
        all_mc_graphs.append((mc,mc_graph))
        print("added and removed edges", added_edges, removed_edges)
        # Make complete graph with respect to ordering
        # Remove edges according to ci_rels
    return all_mc_graphs

