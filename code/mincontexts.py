from itertools import combinations
import time

import networkx as nx

from utils.utils import generate_vals, context_is_contained, parents
            
union = lambda sets : set.union(*sets)

def recursive_search(T_sizes, C, contexts_with_p, val_dict):
    if len(T_sizes) == 1:
        return T_sizes[0]
    
    else:
        mid = len(T_sizes)//2
        candidate_size = T_sizes[mid]
        possible_Ts    = [set(i) for i in combinations(C, candidate_size)]

        not_contained = 0 # to contain possible Ts
        # such that an element in their cartesian product is not in contexts
        for T in possible_Ts:
            vals_T_takes = generate_vals(list(T), val_dict)
            possible_C = set(C).difference(T)
            
            x_T_count=0
            for i, x_T in enumerate(vals_T_takes):
                contained  = context_is_contained(x_T, contexts_with_p,set())
                #print(x_T, vals_T_takes, contexts_with_p, contained)
                if contained:
                    x_T_count += 1
                
                if not contained:
                    not_contained+=1
                    break # To next possible T
                    #return recursive_search(T_sizes[:mid], C, contexts_with_p, val_dict)
                if x_T_count==len(vals_T_takes):
                    return recursive_search(T_sizes[mid:], C, contexts_with_p, val_dict)
            # If none of the Ts of size candidate_size satisfy the criteria, move to next
            # possible T size
            if not_contained==len(possible_Ts):
                return recursive_search(T_sizes[:mid], C, contexts_with_p, val_dict)

    
def binary_minimal_contexts(csi_rels, val_dict, pairwise=True):
    minimal_cs      = set()
    minimal_cs_dict = {}
    #print("beagn with", csi_rels,"\n")
    # TODO edge cases when C is empty

    # We get all the unique pairs involved in all the csi relations with non-empty contexts
    # TODO Think about making pairs a set instead of tuple
    
    mcs   = set()
    #pairs = [A.union(B) for (A,B,S,C) in csi_rels.copy() if len(A)==1 and len(B)==1]
    #pairs_nocopies = []
    #for p in pairs:
    #    if p not in pairs_nocopies:
    #        pairs_nocopies.append(p)
    #triples_nocopies = []
    if pairwise:
        triples = [(A,B,S) for (A,B,S,C) in csi_rels.copy() if len(A)==1 and len(B)==1]
    else:
        triples = [(A,B,S) for (A,B,S,C) in csi_rels.copy()]
    triples_nocopies=[]
    for t in triples:
        if t not in triples_nocopies:
            triples_nocopies.append(t)
    
    for t in triples_nocopies:
        p = t[0].union(t[1])
        con_set = t[2]
        csi_rels_with_p = [c for c in csi_rels.copy() if union([c[0],c[1]])==p and c[2]==con_set]
        contexts_with_p_dups = [set(c[-1]) for c in csi_rels_with_p]
        contexts_with_p = []
        for cc in contexts_with_p_dups:
            if cc not in contexts_with_p:
                contexts_with_p.append(cc)
        context_vars_found = []
        Ts_for_this_pair = []
        done_for_rel = False
        for csi_rel in csi_rels_with_p:
            # Context variables of current CSI relation
            (A,B,S,C1) = csi_rel
            #A = {p[0]}
            #B = {p[1]}
            C = [var for (var,val) in C1]
            #contexts_with_p = [tuple(c[-1]) for c in csi_rels_with_p if c[2]==S]
            #print(csi_rel, contexts_with_p)

            # Possible sizes for T
            possible_T_sizes = [i for i in range(len(C)+1)] #includes empty set and fullset
                            
            # True size of 
            T_size = recursive_search(possible_T_sizes, C, contexts_with_p.copy(), val_dict)
            
            # Find the T
            T_found=False
            if T_size == 0:
                T = []
                possible_Ts = [T]
                debug=possible_Ts.copy()
            elif T_size == len(C):
                T = C.copy()
                possible_Ts = [T]
                debug=possible_Ts.copy()
            else:
                #current_Cs  = [set(c) for (c,_) in minimal_context_dict.keys()]
                possible_Ts = [set(i) for i in combinations(C, T_size)]
                debug=possible_Ts.copy()

                for possible_T in possible_Ts:
                    vals_T_takes = generate_vals(list(possible_T), val_dict)
                    possible_mc = set([c for c in C1 if c[0] not in list(possible_T)])
                    #possible_C   = set(C).difference(possible_T)
                    #print("rel {} T is {}, possible C {}".format(csi_rel,possible_T,possible_mc))
                    
                    x_T_count = 0
                    for i, x_T in enumerate(vals_T_takes):
                        contained  = context_is_contained(x_T, contexts_with_p.copy(),possible_mc)
                        
                        if contained:
                            # print("{} contained for rel {} since contexts are {}".format(x_T, csi_rel, contexts_with_p.copy()))
                            x_T_count +=1
                        if not contained:
                            break
                        if x_T_count==len(vals_T_takes):
                            T = possible_T
                            T_found=True
                            break
                    if T_found:
                        break                            

            if possible_Ts==[]:
                continue
            #print("For rel {}, C is {}, chosen T {} from {}".format(csi_rel, C, T, possible_Ts))

            context_of_rel   = csi_rel[-1]
            context_vars     = [var for (var,val) in context_of_rel]
            new_context_vars = set(context_vars).difference(T)
            for s in context_vars_found:
                assert new_context_vars.intersection(set(T).difference(s))==set()

            
            context_vars_found.append(new_context_vars)

            #not_possible_Ts = 
            
            minimal_context = tuple(tuple(c) for c in context_of_rel if c[0] not in T)
            
            #print("minimal context is ", minimal_context)
            A = t[0]
            B = t[1]
            #(A,B) = p

            if minimal_context == ():
                cond_set = S.union(set(T))
            else:
                cond_set = S.union(set(T).difference(set(C)))

            #print("new cond set", cond_set)

            ci_rel_remaining = (A,B,cond_set)
            #print("remaing, ", ci_rel_remaining, "\n")
            #print("Adding minimal context {} ffrom rel {} possible Ts were {} from t size {}".format(minimal_context, csi_rel, debug, T_size))

                            
            if minimal_context in minimal_cs_dict.keys():
                if ci_rel_remaining not in minimal_cs_dict[minimal_context]:
                    minimal_cs_dict[minimal_context].append(ci_rel_remaining)
            else:
                minimal_cs_dict[minimal_context] = [ci_rel_remaining]


    for rel in csi_rels:
        (A,B,S,Ci) = rel
        for mincontext in list(minimal_cs_dict.keys()):
            if set(Ci) == set(mincontext) and (A,B,S) not in minimal_cs_dict[mincontext]:
                minimal_cs_dict[mincontext].append((A,B,S))
    print(csi_rels)
    print("\n\n\n\n")
    print(minimal_cs_dict, "\n\n\n\n")
    return minimal_cs_dict


def minimal_context_dags(order, csi_rels, val_dict, mec_dag=None, closure=None):
    mc_time = time.time()
    minimal_contexts=binary_minimal_contexts(csi_rels, val_dict)
    minimal_context_dags = []
    print("minimal contexts are", list(minimal_contexts.keys()))
    #minimal_contexts[()]+=extra_rels


    
    for minimal_context, ci_rels in minimal_contexts.items():
        print(ci_rels)
        
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

                
                
                minimal_context_dag.add_edge(pi_i,pi_j)

                conditioning_set = set(remaining_vars[:j]).difference({pi_i,pi_j})
                #print("checking ", pi_i,pi_j, conditioning_set)

                #print("added edge {}, cond set is {}".format((pi_i,pi_j), conditioning_set))

                for ci_rel in ci_rels:
                    A  = ci_rel[0]
                    B  = ci_rel[1]
                    Ci = ci_rel[2]
                    #if A.union(B) == {pi_i,pi_j} and Ci.issubset(conditioning_set):
                        #print("MUST REMOVE ",pi_i,pi_j)

                    if A.union(B) == {pi_i,pi_j} and Ci == conditioning_set:
                        if minimal_context_dag.has_edge(pi_i,pi_j):
                            minimal_context_dag.remove_edge(pi_i,pi_j)
                            minimal_context_dag.add_nodes_from([pi_i, pi_j])
                            
        #print("order is, ",order)
        
        if mec_dag:
            for edge in minimal_context_dag.edges:
                order_printed=False
                if edge not in mec_dag.edges:
                    ci_rels_w_vars = [rel for rel in ci_rels if rel[0].union(rel[1]) == {edge[0]}.union({edge[1]})]
                    if not order_printed:
                        #print(order)
                        order_printed=True
                        #print(parents(mec_dag,edge[0]), parents(mec_dag, edge[1]))
                        #print(edge, "\ncirels w var\n", ci_rels_w_vars, "\nall mc ci rels\n",ci_rels,"\ncsi rels from graphoid\n", csi_rels,"\nfrom tree\n", closure)
        
        print("Adding mc {} and mcdag nodes {} edges {}".format(minimal_context, minimal_context_dag.nodes,minimal_context_dag.edges))
        minimal_context_dags.append((minimal_context, minimal_context_dag))
    return minimal_context_dags
