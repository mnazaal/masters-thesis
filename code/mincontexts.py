from itertools import combinations, permutations
import time

from networkx.readwrite.graph6 import data_to_n
from utils.utils import undirected_both, coming_in, v_structure, data_to_contexts
from gsq.ci_tests import ci_test_bin, ci_test_dis
import matplotlib.pyplot as plt

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

    
def binary_minimal_contexts(csi_rels, val_dict, pairwise=False):
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

vars_of_context = lambda context:  [] if context==[()] else  [var for (var,val) in context] 

def minimal_context_dags_1(min_contexts, dataset, alpha=0.01):
    # mincontext DAGs assuming we are given the minimalcontext
    # TODO check if binary data, set CI test accordingly    
    indep_test = ci_test_dis


    minimal_context_dags = []
    m = len(min_contexts)
    n,p = dataset.shape

    # Getting the skeletons
    for mc in min_contexts:
        if mc==[()]:
            mc=()
        CI = []
        C = vars_of_context(mc)                         #C
        vars = [i+1 for i in range(p) if i+1 not in C]  #[p]\C
        dag  = nx.DiGraph(nx.complete_graph(vars))
        for j in range(len(vars)+1):
            for k,l in combinations(vars, 2): #For each edge in the complete graph 
                neighbours = list(set(list(dag.neighbors(k))).difference({k,l})) 
                if len(neighbours)>=j:
                #S_sizes = range(len(neighbours)+1)
                #for subset_size in S_sizes:
                    subsets = [set(i) for i in combinations(neighbours, j)]
                    zero_indexed_subsets = [set([i-1 for i in s]) for s in subsets]
                    #print(k-1,l-1,zero_indexed_subsets)
                    for subset in zero_indexed_subsets:
                        # For the CI testing, we need zero indexing
                        p_val = indep_test(data_to_contexts(dataset, mc), k-1, l-1, subset)
                        if p_val > alpha:
                            dag.remove_edges_from([(k,l),(l,k)])
                            if ({k},{l}, subset) not in CI:
                                CI.append(({k},{l}, subset))
                            if ({l},{k}, subset) not in CI:
                                CI.append(({l},{k}, subset))
        minimal_context_dags.append((mc, dag, CI))
    print(minimal_context_dags)
    P = []

    # Orienting edges and getting a graph P with directed edges
    for i in range(m):
        mc  = minimal_context_dags[i][0]
        dag = minimal_context_dags[i][1]
        CI  = minimal_context_dags[i][2]
        for edges in combinations(list(dag.edges), 3):
            for s in dag.nodes:
                s_neighbours = dag.neighbors(s)
                for k,l in combinations(s_neighbours, 2):
                    if (k,l) not in dag.edges and (l,k) not in dag.edges: # Making sure its an unshielded triple
                    #print("induced graphs",k,s,l)
                        for (K,L,S) in CI:
                            if {k}==K and {l}==L and s not in S:
                                # Here, the edges l<->s<->k and l->s<-k  become 
                                # remove the undirected edges
                                
                                # We only orient if
                                
                                # adding k->s
                                # First check if we do not have s->k already in P, since otherwise it creates a cycle
                                if not ((s,k) in P or (s,l) in P): # If either of the edges has already been oriented, we skip this v-structure
                                    if (s,k) in list(dag.edges):
                                        dag.remove_edge(s,k)
                                        P.append((k,s))
                                # adding l->s
                                    if (s,l) in list(dag.edges):
                                        dag.remove_edge(s,l)
                                        P.append((l,s))
                                #dag.remove_edges_from([(k,s),(s,k),(s,l),(l,s)]) # redundant, can remove few edges here
                                #dag.add_edges_from([(k,s),(l,s)])
                                #print("orienting", (k,s),(l,s), "because", K,L,S)
                                #P.append((k,s))
                                #P.append((l,s))
    dag_P = nx.DiGraph()
    dag_P.add_nodes_from([i+1 for i in range(p)])
    dag_P.add_edges_from(P)
    
    # Getting a valid ordering
    possible_orderings = nx.all_topological_sorts(dag_P)

    ordering_found=False
    for order in possible_orderings:
        if not ordering_found:
            minimal_context_dags_ordered = []
            new_v_struct=False
            for i in range(m):
                mc  = minimal_context_dags[i][0]
                dag = minimal_context_dags[i][1]
                CI  = minimal_context_dags[i][2] 

                # edges in the fully connected DAG with this order
                # Below line relies on how "combinations" works, namely,
                # e.g. given [a,b,c,d] it returns [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                es = [(i,j) for i,j in combinations(order, 2) if i<j]

                for (u,v) in es:
                    if (u,v) in dag.edges and (v,u) in dag.edges:
                        # direct the edge u<->v as u->v by deleting u<-v
                        dag.remove_edge(v,u)

                        # check if this introduces a v-structure
                        edges_in = coming_in((u,v), dag)

                        # even if we get 1 new v-structure we move on to the next ordering
                        new_v_struct = any(list(map(lambda e: v_structure((u,v),e,dag), edges_in)))
                        if new_v_struct:
                            break
            
            if new_v_struct:
                break
            # if no new vstruct, we choose this ordering
            # minimal_context_dags_ordered.append((mc, dag))
            ordering_found=True
    
    if ordering_found:
        print("Chosen ordering is", order)
    if not ordering_found:
        print("Ordering not found")

    for mc, dag, _ in minimal_context_dags:
        dag_edges = list(dag.edges)
        removed=[]
        for (u,v) in dag_edges:

            if (v,u) in dag_edges and (u,v) not in removed and (v,u) not in removed:
                order_v, order_u = order.index(v), order.index(u)
                if order_v<order_u:
                    dag.remove_edge(u,v)
                    removed.append((u,v))
                if order_u<order_v:
                    dag.remove_edge(v,u)
                    removed.append((v,u))
    
    for mc,dag,_ in minimal_context_dags:
        plt.figure()
        #node_dict = {1:1, 2:2, 3:3, 4:5, 5:6}
        #dag = nx.relabel_nodes(dag, lambda x: node_dict[x])
        dag_pos = nx.drawing.layout.shell_layout(dag)
        nx.draw(dag, pos=dag_pos, with_labels=True)
    

