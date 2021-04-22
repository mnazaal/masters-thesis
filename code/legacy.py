#======================================================
# models.py
#======================================================
# Classes to store variables, contexts etc on nodes
# of the CSTree
#======================================================

from typing import NewType, Union


# We need to separate the values and the variables because
# we need it for the context
class Variable(object):
    def __init__(self, name: str, vals: tuple[str]):
        self.name = name
        self.vals = vals
        
    def __eq__(self, other):
        return self.name == other.name and self.vals == other.vals
        
    def __str__(self):
        return "Variable("+self.name+")"
    
    def __repr__(self):
        return "Variable("+self.name+")"


class Context(object):
    def __init__(self, var: Variable, val: str):
        # Check if value is possible in variable
        self.var = var
        self.val = val
        
    def __eq__(self, other):
        return self.var == other.var and self.val == other.val
    
    def __str__(self):
        return "Context<{},{}>".format(self.var.name, self.val)
    
    def __repr__(self):
        return "Context<{},{}>".format(self.var.name, self.val)

    
class CiRelation(object):
    def __init__(self, vs: tuple[Variable], s: tuple[Variable]):
        # check if everything disjoint
        # check if tuple has only 2 elements
        # check if maybe better to use sets than tuples
        self.vs = vs
        self.s  = s
        
    def __eq__(self, other):
        pass
        
    def __str__(self):
        return "(" + self.vs[0].name + " ⊥ " + self.vs[1].name + " | " + str(self.s.name) + ")"
    
    def __repr__(self):
        return "(" + self.vs[0].name + " ⊥ " + self.vs[1].name + " | " + str(self.s.name) + ")"


class Distribution(object):
    def __init__(self, var : Variable, cs: Union[tuple[Context],Context]):
        self.var = var
        self.cs  = cs
        
    def to_nodes(self):
        # Assuming var is of the form Xi
        # and the contexts are of the class Context 
        return (self.var.name[1:], [c.val for c in self.cs])
        
    def __eq__(self, other):
        if isinstance(other, Distribution):
            return self.var == other.var and self.cs == other.cs
    
    def __hash__(self):
        return hash(str(self))
    
        
    def __str__(self):
        if isinstance(self.cs, tuple):
            return "P("+self.var.name+ " | "+ ''.join(str(c.var.name)+"="+str(c.val)+", " for c in self.cs)+")"
        else:
            return "P("+self.var.name+ " | "+ str(self.cs.var.name)+"="+str(self.cs.val)+", " +")"
    
    def __repr__(self):
        if isinstance(self.cs, tuple):
            return "P("+self.var.name+ " | "+ ''.join(str(c.var.name)+"="+str(c.val)+", " for c in self.cs)+")"
        else:
            return "P("+self.var.name+ " | "+ str(self.cs.var.name)+"="+str(self.cs.val)+", " +")"    # define equality, useful for colouring


        
#======================================================
# mincontexts.py
#======================================================
# Finding minimal contexts
#======================================================

from itertools import combinations
import time

import networkx as nx

from utils import generate_vals, context_is_contained


def minimal_contexts(csi_rels, val_dict):
    mcs = []
    # We get all the unique pairs involved in all the csi relations with non-empty contexts
    # TODO Think about making pairs a set instead of tuple
    pairs = set((tuple(c[0]) for c in csi_rels))
    
    # Set for minimal contexts
    minimal_cs      = set()
    # Dict for minimal contexts
    # The keys are the minimal contexts
    # The values are the CI relations that hold in that minimal context
    minimal_cs_dict = {}
    
    # For each pair p
    for p in pairs:
        
        # Get the csi relations involving p
        csi_rels_with_p  = [c for c in csi_rels if tuple(c[0])==p]            
        
        # Get the contexts for each of the csi relations above
        contexts_under_p = [c[1] for c in csi_rels_with_p]
        
        print("Pair is ", p, "rels with p", csi_rels_with_p)
        
        # For each of the csi relations we have for pair p
        for rel in csi_rels_with_p:
            skips=0
            # Get the variables involved in the context of the csi relation rel
            C = [i for (i,j) in rel[1]]
            
            # For each of the variables involved, we generate all possible subsets in decreasing order
            # For each subset of the variables T, we generate the cartesian products containing all possible values T can take
            # From this cartesian product, we check whether we have a subset T where all values in the cartesian product are indeed contained
            # in the csi relations containing the pair p
            
            # If we find such a T, then the context in rel for the variables in C\T form a minimal context

            # for all possible subsets T for these variables
            #print(p)
            for subset_size in range(len(C),-1,-1):
                
                subsets = [set(i) for i in combinations(C, subset_size)]
                
                #print(subsets)
                #print("SUBSETS", subsets)
                # We are guaranteed that if for example i,j are contained, there is no k such that i,j,k is contained since we
                # check all subsets of size 3 before that
                T_found=False
                for T in subsets:
                    # all possible values T can take
                    # TODO make this a generator
                    vals_T_takes = generate_vals(list(T), val_dict)
                    
                    # Empty set means this list has size 1, since it is [[]]
                    
                    # We make a copy so we can remove contexts we find in the discovered CSI relations
                    # This reduces time taken to check if the next possible T configuration exists in the discovered CSI relations
                    contexts_under_p_copy = contexts_under_p.copy()
                    T_counter = 0
                    for i in range(len(vals_T_takes)):
                        contained = context_is_contained(vals_T_takes[i], contexts_under_p_copy)
                        if not contained:
                            skips+=1
                            #print("skipped", skips)
                        if not contained and len(vals_T_takes) != 1:
                            break
                        if i==len(vals_T_takes)-1:
                            # To put empty contexts first
                            T_found=True
                            mc = tuple(i for i in rel[1] if i[0] in C and i[0] not in list(T))
                            #print("CSI Rels with p are ", csi_rels_with_p, "with Minimal Context ", mc, "\n")
                            # Variables T for which X_p0 _||_ X_p1 | X_T, mc
                            vs_besides_mc = set(T).difference(set(p))

                            if mc in minimal_cs_dict:
                                minimal_cs_dict[mc].append((set(p), vs_besides_mc))
                            else:
                                # List containing tuples of the form (p, vars) where the pair p is independent conditioned on vars in the context mc
                                # p, vars are sets
                                minimal_cs_dict[mc]= [((set(p), vs_besides_mc))]
                            break
                            
                            
                        # The moment we know that the current val is not a subcontext of any contexts under p, we move onto next subset T

                        # If we know this particular value for T exists in the discovered CSI relations, we remove
                        # it from these relations which makes it a bit faster to search for the next particular T value
                        if contained:
                            contexts_under_p_copy.remove(contained)
                        # If we make it through all possible values, we need to separate out T


                        #minimal_cs.add(mc)
                # If we did find a T, we do not need to check the rest of the subsets of the same size
                # (since if there was another subset of the same size which also gives us a minimal context,
                # we would have found it when the subset size was larger)
                if T_found:
                    break
            # If we did find a T we need not checking smaller subset sizes
            if T_found:
                # Add the minimal context
                break
                # TODO remove duplicates from list within loop structure

    # Reduce looping instead of making minimal contexts unique separately
    #minimal_cs.sort()
    #return list(minimal_cs for minimal_cs,_ in itertools.groupby(minimal_cs))
    return minimal_cs_dict


def binary_minimal_contexts(csi_rels, val_dict):
    mcs = []
    # We get all the unique pairs involved in all the csi relations with non-empty contexts
    # TODO Think about making pairs a set instead of tuple
    pairs = set((tuple(c[0]) for c in csi_rels))
    
    # Set for minimal contexts
    minimal_cs      = set()
    # Dict for minimal contexts
    # The keys are the minimal contexts
    # The values are the CI relations that hold in that minimal context
    minimal_cs_dict = {}
    
    # For each pair p
        
    for p in pairs:
        # Get the csi relations involving p
        csi_rels_with_p  = [c for c in csi_rels if tuple(c[0])==p]            
        
        # Get the contexts for each of the csi relations above
        contexts_under_p = [c[1] for c in csi_rels_with_p]
        #print("Pair is ", p, "rels with p", csi_rels_with_p)
        # For each of the csi relations we have for pair p
        done_for_rel = False
        for rel in csi_rels_with_p:
            skips=0
            
            # Get the variables involved in the context of the csi relation rel
            C = [i for (i,j) in rel[1]]
            
            # For each of the variables involved, we generate all possible subsets in decreasing order
            # For each subset of the variables T, we generate the cartesian products containing all possible values T can take
            # From this cartesian product, we check whether we have a subset T where all values in the cartesian product are indeed contained
            # in the csi relations containing the pair p
            
            # If we find such a T, then the context in rel for the variables in C\T form a minimal context

            # for all possible subsets T for these variables
            #print(p)
            #mid_point        = int(len(C)/2)
            possible_T_sizes = [i for i in range(len(C)+1)]
            
            T_found=False
            while not T_found:
                midpoint    = int(len(possible_T_sizes)/2)
                T_size      = possible_T_sizes[midpoint]
                possible_Ts = [set(i) for i in combinations(C, T_size)]
                
                # For loop 1
                for T in possible_Ts:
                    
                    T_in         = False
                    vals_T_takes = generate_vals(list(T), val_dict)
                    
                    contexts_under_p_copy = contexts_under_p.copy()
                    
                    # For loop 2
                    x_T_contained=0
                    for i, x_T in enumerate(vals_T_takes):
                        if possible_T_sizes == [len(C)]:
                            search_upperhalf = False
                            search_lowerhalf = False
                            break

                        #print("T length and xT length", T_size, x_T)
                        contained  = context_is_contained(x_T, contexts_under_p_copy)
                        #print(p, possible_T_sizes, len(C)-possible_T_sizes[0]
                        
                        # Case T={} is always in 
                        #if len(vals_T_takes)==0:
                        #    T_in  = True
                        #    search_upperhalf = True
                        #    break # For loop 2
                            
                        # Case T non empty
                        # Case we find a non empty T such that all x_T is in
                        #search_upperhalf,search_lowerhalf=False,False
                        if contained:
                            x_T_contained += 1
                            contexts_under_p_copy.remove(contained)
                            search_upperhalf = False

                            # If the number of outcomes contained is the same as the number of possible outcomes
                            # Then we found a T such that X_T=x_T is inside all the relations
                            if x_T_contained == len(vals_T_takes):
                                T_in = True
                                #possible_T_sizes = possible_T_sizes[midpoint:]
                                search_upperhalf = True
                                search_lowerhalf = False
                                break # For loop 2
                                # TODO break here? but technically it 

                        # If we do find a context 
                        if not contained:
                            skips+=1
                            # TODO THink about len!=1
                            # If we find an outcome x_T not contained, we move onto next T
                            #print("Skipped", skips)
                            search_lowerhalf = True
                            search_upperhalf = False
                            #possible_T_sizes = possible_T_sizes[:midpoint]
                            break #For loop 2
                            # We found and x_T not in the CSI relations

                    # If we found a T such that all x_T is contained and there was only one remaining possible T size
                    # Then T is the largest set such that C\T is a minimal context
                    if (T_in and len(possible_T_sizes)==1) or possible_T_sizes == [0] or possible_T_sizes == [len(C)]:
                        if possible_T_sizes==[len(C)]:
                            pass
                            #print("T full size", possible_T_sizes)
                        done_for_rel = True
                        T_found        = True # Breaks out of while loop
                        mc = tuple(i for i in rel[1] if i[0] in C and i[0] not in list(T))
                        vs_in_mc = [i for (i,j) in mc]
                        vs_besides_mc = set(T).difference(set(vs_in_mc))

                        if mc in minimal_cs_dict:
                            minimal_cs_dict[mc].append((set(p), vs_besides_mc))
                        else:
                                # List containing tuples of the form (p, vars) where the pair p is independent conditioned on vars in the context mc
                                # p, vars are sets
                            minimal_cs_dict[mc]= [((set(p), vs_besides_mc))]
                        break # For loop 1
                        
                        
                    #if done_for_rel:
                    #    break
                        

                        
                    else:    
                        if search_upperhalf:
                            possible_T_sizes = possible_T_sizes[midpoint:]
                            break # For loop 1
                        if search_lowerhalf:
                            possible_T_sizes = possible_T_sizes[:midpoint]
                            break # For loop 1
                        if T_in:
                            pass
                            #break 

                
                # Technically no need since while loop breaks out when T_found=True
                if T_found:
                    break
            if T_found:
                break
            
    return minimal_cs_dict



def minimal_context_dags(order, csi_rels, val_dict, extra_rels=None):
    # Csi rels are of the form tuple containing pair and list containing context
    mc_time = time.time()
    mcs=binary_minimal_contexts(csi_rels, val_dict)
        
    print("YOO minimal contexts took ", time.time()-mc_time)
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
                
                """
                
                # MC DAGs Case b
                #if nx.d_separated(mc_graph, {pi_i}, {pi_j}, conditioning_set):
                    
                # MC DAGs Case c
                if (set([pi_i, pi_j]),conditioning_set) in ci_rels:
                    #print("ci rels are ", ci_rels)
                    #print("pi_i,pi_j | cond set is ", set([pi_i, pi_j]),conditioning_set)
                    mc_graph.remove_edge(pi_i, pi_j)
                """
                """
                # MC DAGs Case d
                for ci_rel in ci_rels:
                    ci_rel_condset = ci_rel[1]
                    if ci_rel_condset.issubset(set(mc_graph.nodes)):
                        if nx.d_separated(mc_graph, {pi_i}, {pi_j}, ci_rel_condset):
                            if mc_graph.has_edge(pi_i,pi_j):
                                mc_graph.remove_edge(pi_i, pi_j)
                """
                        
                
                
        all_mc_graphs.append((mc,mc_graph))
        print("added and removed edges", added_edges, removed_edges)
        # Make complete graph with respect to ordering
        # Remove edges according to ci_rels
    return all_mc_graphs



#======================================================
# cstree.py
#======================================================
# Generating CSTrees
#======================================================


def dag_to_cstree(val_dict, ordering=None, dag=None, construct_last=False):
    """
    =============================================================================
    description:
    =============================================================================
    Construct a CSTree. If given an ordering but no DAG, the tree is not coloured. 
    If no causal ordering is given, we take any 
    causal ordering that respects the DAG
    
    =============================================================================
    inputs:
    =============================================================================
    - val_dict  :: {Int:[Int]}
    
    Dictionary containing variables as keys and values a variable can take as
    values
    
    - ordering    :: [int]
    
    Ordering of the nodes to consider. If not given, some topological sort of the 
    DAG is given, gives error if both ordering and DAG is not given
    
    - dag         :: nx.DiGraph()
    
    DAG which we turn into a CSTree
    
    =============================================================================
    outputs:
    =============================================================================
    - cstree      :: nx.DiGraph()
    
    CSTree constructed from the DAG
    
    - stages :: {str:[[(int,int)]]}
    
    Dictionary containing a colour (in hex string) and list of nodes (a node is 
    a list of int,int tuples) for each non-singleton stage
    
    - colour_scheme :: {[(int,int)]:str}
    
    Dictionary containing node,colour pairs
    
    =============================================================================
    thoughts:
    =============================================================================
    Constructing levels is not necessarily a sequential task if we know the 
    ordering and the values each variable can take like we have here, think
    of a way to do it using threading/multiprocessing.
    Also think of maybe storing nodes per level as a generator
    Also create a map function for first task
    This being said the memory limit reaches before time limit
    TODO make plot even for last level
    TODO Check if ordering is consistent with DAG if both given
    """
    
    if dag is None and ordering is None:
        raise ValueError("If no ordering is given a DAG must be provided")
    
    # If DAG given but ordering is not, take some topological ordering
    if dag and ordering is None:
        ordering = dag_topo_sort(dag)
    
    # Initialize empty graph for CStree
    cstree = nx.DiGraph()
    
    # Initialize dictionary to store colours of non-singleton stages
    stages        = {}
    colour_scheme = {}
    
    # Create first root node
    # This is a tuple so the Cartesian product makes sense later
    roots = (("Root",),)

    #if not construct_last:
    #    ordering = ordering[:-1]
    level=0
    for var in ordering[:-1]:
        # Level of the current variable
        level+=1
        
        # Values current variable can take
        vals = val_dict[var]
        
        # Nodes in current level
        current_level_nodes = tuple([(var, val) for val in vals])
        
        # If we are at the first variable, construct the Root node
        if ordering.index(var)==0:
            edges = [("Root", (n,)) for n in current_level_nodes]
        # Otherwise chain from previous roots
        else:
            edges = [[(r, r+(n,)) for r in roots] for n in current_level_nodes]
            edges = list(chain.from_iterable(edges))
        cstree.add_edges_from(edges)
        
        # These are the nodes in the current level which will be roots for next level
        roots = [j for (i,j) in edges]
        
        # Encoding CI rels if DAG is given
        #if dag and ordering.index(var)==0:
            
        
        
        if dag:
            # TODO Delete below
            # π_k+1
            next_var       = ordering[ordering.index(var)+1]
            preceding_vars = ordering[:ordering.index(var)-1]
            
            
            # Parents_G(π_k+1)
            pars           = parents(dag, next_var)
            
            
            # TODO Put a condition to stop the case where we have no full cartesian prodict
            # i guess its the len(sc)!=level below
            #if len(pars)==level:
            #    print("we got parents same as variables, mistake somewher")
            #    pars=[]
            #if pars == [] and ordering.index(var)==0:
            #    stage_contexts = []
            #else:
            stage_contexts=generate_vals(pars,val_dict)
            
            
            for sc in stage_contexts:
                # TODO Faster method to partition nodes in current level into stages
                # TODO IMPORTANT, previously sc!=[] was also included in below predicate, WHY?
                #if len(sc)!=level: # this was to prevent singleton stages to be added i believe
                stage_nodes = [n for n in roots if set(sc).issubset(n) and len(sc)!=level]
                colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                if stage_nodes != []: # In root level its empty
                    stages[colour] = stage_nodes

                    for node in stage_nodes:
                        colour_scheme[node]=colour
            

            # Description of colouring tree from DAG
            # get the variable after var, v_i+1
            # get parents of v_i+1 in DAG, call it Parents(v_i+1)
            # get the variables (1,2,..,v_i)|Parents(v_i+1) which is
            # the intersection of the variables before v_i+1 in ordering and Parents(v_i+1), call this set of variables C 
            # All nodes in this level represent p(v_i+1|C)
            # Which means we need to enumerate all possible values C can take and label nodes accordingly
            # Also remember to add to colour_dict
            

    return cstree, stages, colour_scheme



def cstree_pc(dataset, pc_method="pc", val_dict=None, ordering=None, draw=True, draw_all_mec_dags=False,draw_mec_dag=False,draw_tree=False,test="epps", use_dag = True):
    # Get number of samples n and dimension of features p
    n,p = dataset.shape
    
    # At this point the variables are ordered according to the other in the pandas dataframe
    if val_dict is None:
        val_dict = generate_state_space(dataset)
        for outcomes in val_dict.values():
            if len(outcomes)<2:
                raise ValueError("At least one random variable is taking only 1 value in the data")
        
    if pc_method == "pc":
        # If the data is binary we do a different test in the PC algorithm
        binary_data = True if all(list(map(lambda f: True if len(f)==2 else False, list(val_dict.values())))) else False
        

        # Set the test to get CPDAG
        if binary_data:
            pc_test = ci_test_bin
        else:
            pc_test = ci_test_dis
            

        # Get CPDAG skeleton
        (g, sep_set) = estimate_skeleton(indep_test_func=pc_test,
                                         data_matrix=dataset,
                                        alpha=0.05)

        # Get the CPDAG
        cpdag = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        cpdag = nx.relabel_nodes(cpdag, lambda x: x+1)
    
    elif pc_method=="pgmpy":
        cpdag_model = PC(pd.DataFrame(dataset, columns=[i for i in range(1,dataset.shape[1]+1)]))
        cpdag_pgmpy = cpdag_model.estimate(return_type="cpdag")
        cpdag = nx.DiGraph()
        cpdag.add_nodes_from([i for i in range(1, dataset.shape[1]+1)])
        cpdag.add_edges_from(list(cpdag_pgmpy.edges()))
        
    #dags_bn = cpdag_to_dags(cpdag.copy())
    cpdag =  pdag.PDAG(nodes=cpdag.nodes,edges=cpdag.edges )
    
    
    dags_bn = []
    all_arcs = cpdag.all_dags()
    for dags in all_arcs:
        temp_graph = nx.DiGraph()
        temp_graph.add_edges_from(dags)
        temp_graph.add_nodes_from([i for i in range(1, dataset.shape[1]+1)])
        dags_bn.append(temp_graph)
        
        if draw_mec_dags:
            fig=plt.figure(figsize=(5,5))
            options = {
                        'node_color': 'white',
                        'node_size': 1000,
                    }

            nx.draw_networkx(temp_graph,pos=nx.drawing.layout.shell_layout(temp_graph), **options)
            ax = plt.gca()
            ax.collections[0].set_edgecolor("#000000")
            plt.show()




    # Get all the DAGs in the MEC
    found_dag = False
    if ordering:
        for dag in dags_bn:
            all_orders = list(nx.all_topological_sorts(dag))
            if ordering not in all_orders:
                continue
            else:
                found_dag = True
                break
        if found_dag:
            dags_bn   = [dag]
        else:
            raise ValueError("No DAG in MEC of CPDAG with this ordering")
    cstree_count=0
    
    for mec_dag_num, mec_dag in enumerate(dags_bn):
        if ordering and found_dag:
            orderings=[ordering]
        else:
            orderings=nx.all_topological_sorts(mec_dag)

        #print("testing cstreepc under ordering")
        # Plot MEC DAG
        fig = plt.figure()
        fig.suptitle("MEC DAG number {}".format(mec_dag_num), fontsize=13)
        nx.draw(mec_dag,pos=nx.drawing.layout.shell_layout(mec_dag),with_labels=True)
        plt.show()

        for order_num, order in enumerate(orderings):
            cstree_count+=1

            time_for_wholetree=time.time()

            # Generate CSTree from DAG
            t1=time.time()
            if not use_dag:
                mec_dag = None
            tree,stages,cs = dag_to_cstree(val_dict, order, mec_dag)
            #print("Time taken to generate CSTree from MEC DAG {}, ordering number {} is {}".format(mec_dag_num, order_num, time.time()-t1))


            # csi_rels is the actual csi rels from tree
            # old style is the one where we split the relations into pairs with fixed contex
            csi_rels, oldstylecsi_rels = stages_to_csi_rels(stages, order)

            # Coloured CStree after conversion from DAG
            draw_dagtocstree = False
            if draw_dagtocstree:
                node_colours = [cs.get(n, "#FFFFFF") for n in tree.nodes()]
                if p<6:
                    pos = graphviz_layout(tree, prog="dot", args='')
                else:
                    pos =  graphviz_layout(tree, prog="twopi", args="")
                fig=plt.figure(figsize=(10,10))
                nx.draw(tree, node_color=node_colours, pos=pos, with_labels=False, font_color='white',linewidths=1)
                ax= plt.gca()
                ax.collections[0].set_edgecolor("#000000")
                #plt.savefig('savedplots/dagcoloured'+str(cstree_count)+'.pdf')

            # Perform further context-specific tests
            t1=time.time()
            tree, stages, cs = colour_cstree(tree, order, dataset, stages, cs, test=test)
            print("Time taken to learn CSTree from MEC DAG {}, ordering number {} is {}".format(mec_dag_num, 
                                                                                                   order_num, 
                                                                                                   time.time()-t1))

            #print("STAGES AFTER LEARNING TREE", stages)

            # Coloured CStree after performing further context-specific tests
            if draw:
                node_colours = [cs.get(n, "#FFFFFF") for n in tree.nodes()]
                if p<6:
                    pos = graphviz_layout(tree, prog="dot", args='')
                else:
                    pos =  graphviz_layout(tree, prog="twopi", args="")
                fig = plt.figure(figsize=(10,10))
                nx.draw(tree, node_color=node_colours, pos=pos, with_labels=False, font_color='white',linewidths=1)
                ax  = plt.gca()
                ax.collections[0].set_edgecolor("#000000")
                #plt.savefig('savedplots/fullcoloured'+str(cstree_count)+'.pdf')
                
            print("stages are ", stages)


            # csi_rels is the actual csi rels from tree
            # old style is the one where we split the relations into pairs with fixed contex
            csi_rels, oldstylecsi_rels = stages_to_csi_rels(stages, order)
            
            #print(csi_rels, oldstylecsi_rels)

            # For each of the original csi relations
            # we get the pair csi relations by pushing other variables onto the conditioning set
            pair_csi_rels = []
            for c in csi_rels:
                for var in c[1]:
                    pair_csi_rels.append((c[0], {var}, c[1].difference({var}), c[3]))
            tree_dseps = []
            for a in pair_csi_rels:
                if (a[0].union(a[1]), a[2]) not in tree_dseps:
                    tree_dseps.append((a[0].union(a[1]), a[2]))
            mintemp = [({list(c[0])[0]}, {list(c[0])[1]},c[1],[]) for c in tree_dseps]


            #print(csi_rels)
            all_mc_graphs = minimal_context_dags(order, oldstylecsi_rels, val_dict, mintemp)
            mecdag_count=0
            for mc,g in all_mc_graphs:
                mecdag_count+=1
                fig=plt.figure(figsize=(5,5))
                context_string = "Context "+"".join(["X"+str(i)+"="+str(j)+" " for (i,j) in mc]) if mc != () else "Empty context after learning CSI relations"
                fig.suptitle(context_string, fontsize=13)
                #nx.draw(g,  pos=nx.drawing.layout.shell_layout(g), with_labels=True)

                options = {
                    'node_color': 'white',
                    'node_size': 1000,
                }

                nx.draw_networkx(g,pos=nx.drawing.layout.shell_layout(g), **options)
                ax = plt.gca()
                ax.collections[0].set_edgecolor("#000000")
                #plt.savefig('savedplots/mecdagnum'+str(mecdag_count)+'for'+str(cstree_count)+'.pdf')
            if p<10:
                plt.show()

            print("this whole iteration took", time.time()- time_for_wholetree, "s")

            
def stages_to_csi_rels(stages, order):
    
    csi_rels     = []
    oldcsi_rels  = []
    # For each level l
    for var in order[:-1]:
        
        # Get the stages in that level
        #stages_in_l = {k:v for k,v in stages.items() if nx.shortest_path_length(tree, "Root", v[0])==order.index(var)+1}
        
        stages_in_l = {k:v for k,v in stages.items() if len(v[0])==order.index(var)+1}
        print("level ",order.index(var)+1, stages_in_l)
        
        # For each stage in that level
        for k,v in stages_in_l.items():
            # Get the context represented by that stage
            print("stage stocsi adding")
            common_context=context_per_stage(v)
            
            # Get the variables in the context
            context_vars = [i for (i,j) in common_context]
            
            # Get the variables in stage node (current level) but not in context
            other_vars = [i for (i,j) in v[0] if i not in context_vars]
            
            # Get the next variable in causal ordering
            next_var = order[order.index(var)+1]
            
            
            #print("adding ",({next_var}, set(other_vars), set(), common_context))
            for o in other_vars:
            #    print(({next_var, o}, common_context))
                oldcsi_rels.append(({next_var, o}, common_context))


            csi_rels.append(({next_var}, set(other_vars), set(), common_context))
    
    return csi_rels, oldcsi_rels