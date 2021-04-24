from itertools import combinations
import time

import networkx as nx

from utils import generate_vals, context_is_contained, parents
            
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
        

    
def binary_minimal_contexts(csi_rels, val_dict, pairwise=True):
    minimal_cs      = set()
    minimal_cs_dict = {}
    # TODO edge cases when C is empty
    
    
    
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
                (A,B,S,C) = csi_rel
                C = [var for (var,val) in C]
                
                
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

                if minimal_context == ():
                    cond_set = S.union(set(T))
                else:
                    cond_set = S.union(set(T).difference(set(C)))

                #print("new cond set", cond_set)

                ci_rel_remaining = ({A},{B},cond_set)

                                
                if minimal_context in minimal_cs_dict.keys():
                    if ci_rel_remaining not in minimal_cs_dict[minimal_context]:
                        minimal_cs_dict[minimal_context].append(ci_rel_remaining)
                else:
                    minimal_cs_dict[minimal_context] = [ci_rel_remaining]


    for rel in csi_rels:
        (A,B,S,Ci) = rel
        for mincontext in list(minimal_cs_dict.keys()):
            if Ci == mincontext:
                
                minimal_cs_dict[mincontext].append(rel)
                
    return minimal_cs_dict


def minimal_context_dags(order, csi_rels, val_dict, mec_dag, closure=None):
    mc_time = time.time()
    minimal_contexts=binary_minimal_contexts(csi_rels, val_dict)
    minimal_context_dags = []
    #minimal_contexts[()]+=extra_rels


    
    for minimal_context, ci_rels in minimal_contexts.items():
        if closure:
            for rel in closure:
                if set(rel[-1])==set(minimal_context):
                    ci_rels.append((rel[0], rel[1], rel[2]))
        
        minimal_context_dag = nx.DiGraph()
        C = [var for (var,val) in minimal_context]
        remaining_vars = [var for var in order if var not in C]
        nodes = len(remaining_vars)
        for i in range(nodes):
            for j in range(i+1, nodes):
                pi_i = remaining_vars[i]
                pi_j = remaining_vars[j]

                minimal_context_dag.add_nodes_from([pi_i, pi_j])
                
                minimal_context_dag.add_edge(pi_i,pi_j)

                conditioning_set = set(remaining_vars[:j]).difference({pi_i,pi_j})

                for ci_rel in ci_rels:
                    A = ci_rel[0]
                    B = ci_rel[1]
                    Ci = ci_rel[-1]
                    if A.union(B) == {pi_i,pi_j} and Ci.issubset(conditioning_set):
                        if minimal_context_dag.has_edge(pi_i,pi_j):
                            minimal_context_dag.remove_edge(pi_i,pi_j)
                        else:
                            pass
                            #print("did not remove, conditiioning set is ,",conditioning_set, "ci rel is", Ci, Ci.issubset(conditioning_set))
                            
        #print("order is, ",order)
        
        for edge in minimal_context_dag.edges:
            order_printed=False
            if edge not in mec_dag.edges:
                ci_rels_w_vars = [rel for rel in ci_rels if rel[0].union(rel[1]) == {edge[0]}.union({edge[1]})]
                if not order_printed:
                    print(order)
                    order_printed=True
                print(parents(mec_dag,edge[0]), parents(mec_dag, edge[1]))
                print(edge, "\ncirels w var\n", ci_rels_w_vars, "\nall mc ci rels\n",ci_rels,"\ncsi rels from graphoid\n", csi_rels)
        

        minimal_context_dags.append((minimal_context, minimal_context_dag))
    return minimal_context_dags
