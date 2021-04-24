# Python imports
import random
from itertools import chain, combinations
import time
import logging

# Third-party library imports
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import epps_singleton_2samp,anderson_ksamp
from gsq.ci_tests import ci_test_bin, ci_test_dis
from networkx.drawing.nx_agraph import graphviz_layout
from pc import estimate_cpdag, estimate_skeleton
import pgmpy
import pandas as pd
from pgmpy.estimators import PC
from causaldag import pdag
import numpy as np

# Project imports
from utils import contained, flatten, cpdag_to_dags, generate_vals, parents, dag_topo_sort,generate_state_space,shared_contexts,data_to_contexts,context_per_stage
from mincontexts import minimal_context_dags,binary_minimal_contexts
from graphoid import decomposition, weak_union, graphoid_axioms


#logger= logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#file_handler = logging.FileHandler("cstree.log")
#logger.addHandler(file_handler)


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

    """
    logger.debug(" ================ \n Start dag_to_cstree \n =====================")
    logger.debug("Converting DAG to CSTree, DAG: {}, ordering: {}".format("given" if dag else "None", 
                                                                             "given" if ordering else "None"))"""
    
    if dag is None and ordering is None:
        raise ValueError("If no ordering is given a DAG must be provided")
        
    if ordering and dag:
        if len(ordering) != len(list(dag.nodes)):
            raise ValueError("The size of the given ordering and nodes in given DAG do not match")
        if ordering not in list(nx.all_topological_sorts(dag)):
            raise ValueError("The ordering given is not valid for the given DAG")
    
    for var in ordering:
        if var not in list(val_dict.keys()):
            raise ValueError("Variables in ordering and value dictionary do not match")
    
    # If DAG given but ordering is not, take some topological ordering
    if dag and ordering is None:
        ordering = dag_topo_sort(dag)

        
    #logger.debug("DAG with edges {} and ordering {} compatible to create CSTree".format(list(dag.edges), ordering))
    
    
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
        
        #logger.debug("Generating level {} of CSTree for variable {} \n".format(level, var))
        
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
            
            if level==1:
                x_2 = ordering[ordering.index(var)]
            
            # TODO Delete below
            # π_k+1
            next_var       = ordering[ordering.index(var)+1]
            

            
            
            # Parents_G(π_k+1)
            pars           = parents(dag, next_var)
            
                
            
            preceding_vars   = ordering[:ordering.index(var)]
            independent_vars = [i for i in preceding_vars if i not in pars]
            
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
                # special case when 2nd variable has no parents, it implies it is independent of the first variable
                # but independence holds both ways and we must encode this fact for the nodes in the first level
                if level == 1 and pars == []:
                    # Nodes in level 1
                    stage_nodes = [n for n in list(cstree.nodes)[1:] if n[0][0] == ordering[level-1]]# -1 because python indexing

                    
                   # logger.debug("Nodes {} belong to stage with common context {}".format(stage_nodes,sc))

                    colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    if stage_nodes != []: # In root level its empty
                        stages[colour] = stage_nodes

                        for node in stage_nodes:
                            colour_scheme[node]=colour

                if independent_vars!=[]:
                    stage_nodes = [n for n in roots if set(sc).issubset(n)]
                   #logger.debug("Encoding {} _||_ {} | {} \n".format(next_var,independent_vars,sc))

                    # TODO Faster method to partition nodes in current level into stages
                    # TODO IMPORTANT, previously sc!=[] was also included in below predicate, WHY?
                    #if len(sc)!=level: # this was to prevent singleton stages to be added i believe

                   # logger.debug("Nodes {} belong to stage with common context{}".format(stage_nodes,sc))

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




    
def stages_to_csi_rels(stages, ordering):
    
    
    csi_rels     = []
    
    p = len(ordering)
        
    for l in range(1,p+1):
        
        stages_l = {c:ns for c,ns in stages.items() if len(ns[0])==l}
        
        
        for c, nodes in stages_l.items():
            
            if len(nodes)==0:
                raise ValueError("There is a singleton stage added as a non-singleton stage")
            common_context = context_per_stage(nodes)
            X_k                  = {ordering[l]}
            X_C                  = set([var for (var,val) in common_context])
            X_k_minus_one        = set([var for (var, val) in nodes[0]])
            X_k_minus_one_diff_C = X_k_minus_one.difference(X_C)
            
            csi_rels.append((X_k, X_k_minus_one_diff_C, set(), common_context))
    return csi_rels
  
        
        
def colour_cstree(c, 
                  ordering, 
                  data_matrix, 
                  stages          = None,
                  colour_scheme   = None,
                  tol_p_val       = 0.05,
                  return_csi_rels = True,
                  test            = "epps",
                  no_dag          = True):
    # c is the Fal
    # levels is the number of levels in the cstree
    # ordering is the causal ordering being considered
    
    level = 1 #0 is root
    # while level<number of variables -1
    # Get all Distributions d in level i
    # For each d1,d2 pair, perform conditional independence
    # Store useful information that will help to colour nodes rapidly
    # After all nodes are coloured, move on to next level
    colours     = 0
    skipped     = 0
    not_skipped = 0
    new_stages  = 0
    
    if stages is None:
        stages = {}
    
    if colour_scheme is None:
        colour_scheme = {}
        
    less_data_counter = 0
    
    levels = len(ordering)
    
    csi_stages = {}
    
    #print("COLOURING CSTREE STARTED WITH STAGES ", stages)
    
    while level<levels+1:
        # at each level we have a list of lists representing nodes in the same stage        
        
        # Nodes in this level
        nodes_l  = [n for n in c.nodes if nx.shortest_path_length(c, "Root", n)==level]
        stages_l = {c:ns for c,ns in stages.items() if len(ns[0])==level}
        stages_known_l = stages_l.copy()
        
        if len(stages_l) == 1:
            level+=1
            csi_stages.update(stages_known_l)
            continue
        #print("starting colouring level ", level, "with stage count", len(stages_l) )
        
        #print("In level ", level, "nodes are ", nodes)
                
        # v0, generate common contexts starting from the empty context to the full contexts
        
        
        # v1, generate all pairs of nodes in the current level
        # TODO Fix the fact that we have sub contexts as different colours, on a related note,
        # fix the fact that you could have nodes belonging to a larger stage but ignored because you already coloured it
        # for example, 110, 111 might be coloured earlier but later we find out 100 and 101 also belong here...?
        
        coloured=[]
        
        # TODO Create 3 cases here for generator of pairs, randomized sampler, pincer movement
        # TODO Check if nodes are in fact ordered as we imagined
        # For each pair of nodes in this level
        break_counter=0
        for n1,n2 in combinations(nodes_l,2):
            # Cases: Both coloured
            #        |--> (a) Both same colour
            #        |--> (b) Both different colour
            #        Atleast one not coloured
            #        |--> (c) n1 coloured, n2 not coloured
            #        |--> (d) n1 not coloured, n2 coloured
            #        |--> (e) Both not coloured
            
            # Updating the list of coloured nodes after each pair
            # Can be done only if merge happens but then must repeat
            # code since it must be done before first test happens anyways
            if colour_scheme.values():
                coloured = list(colour_scheme.keys())
            
            
            # Case (a) : both coloured and same colour
            if (n1 in coloured and n2 in coloured) and (colour_scheme[n1] == colour_scheme[n2]):
                skipped += 1
                continue
                    
            # Case (b,c,d,e) : Other than above
            else:
                # We do not skip the test if 2 nodes do not have the same colour
                not_skipped+=1
                colour_n1 = colour_scheme.get(n1, None)
                colour_n2 = colour_scheme.get(n2, None)
                
                #print("colours are ", colour_n1,colour_n2)
                
                
                # TODO Abstract this part away
                common_c = shared_contexts(n1[:-1],n2[:-1])

                var = ordering[level]
                                
                data_n1 = data_to_contexts(data_matrix,n1,var)
                data_n2 = data_to_contexts(data_matrix,n2,var)

                # Possibly ignore the test if the samples are imbalanced
                avg_data = 0.5*(len(data_n1)+len(data_n2))
                skewed_data=False
                if len(data_n1)< 0.4*avg_data or len(data_n2)< 0.5*avg_data:
                    skewed_data=True
                    #same_distr=False

                if test=="epps":
                    # If we have enough data, do the test
                    if len(data_n1)<5 or len(data_n2)<5 or skewed_data:
                        less_data_counter +=1
                        same_distr=False
                    else:
                        t,p = epps_singleton_2samp(data_n1, data_n2)
                        if p>tol_p_val or p is float("NaN"):
                            same_distr=True
                        else:
                            same_distr=False
                if test=="anderson":
                    if len(data_n1)<5 or len(data_n2)<5 or skewed_data or len(np.unique(data_n1))==1 or len(np.unique(data_n2))==1:
                        less_data_counter +=1
                        p=0
                        same_distr=False
                    else:
                        statistic, critical_vals, p = anderson_ksamp([data_n1, data_n2])
                        if p>tol_p_val:
                            same_distr=True
                        else:
                            same_distr=False
                else:
                    same_distr=False
                    # TODO Think of why this had to be here,
                    # gave error on synthetic set otherwise
                        

                if same_distr:
                    
                    if common_c == []:
                        num_empty_contexts+=1
                        #print("added empty context, total now ", num_empty_contexts, level)
                        #print("empty context data looked at was ", data_n1,data_n2)
                    
                    #print("p({}|{})".format(var, common_c))
                    #print("Merging node colours ", colour_n1, colour_n2)
                    # Nodes below belong to the same stage defined by the common context

                    # To keep track of how many stages we add after colouring from DAG
                    new_stages+=1

                    # These are the new nodes that belong to the stage represented by the common context
                    new_nodes = [n for n in nodes_l if set(common_c).issubset(set(n))]
                    #print("\n new nodes are", new_nodes, "with common context", common_c, "\n")
                    
                    #print("length of level", 2**level, "length of new nodes ", len(new_nodes), "len of context ", len(common_c))
                    #print("common context is", common_c)
                    #print("level ", level, "stages before ", stages)
                    # Keep only stages if any nodes in a stage has this common context
                    #print("ns first item", len(stages_l))
                    
                    stages_remaining = {c:ns for c,ns in stages_l.items() if not set(common_c).issubset(set(ns[0]))}
                    #print("Level ", level, "remaining stages after adding this new stage is ", len(stages_remaining))
                    stages_known_l = stages_remaining.copy()
                    #print("level ", level, "stages remaining ", stages)

                    # Then add the new stage
                    stage_added = False
                    not_added=0
                    while not stage_added:
                        not_added+=1
                        # Just to make sure they are different colours
                        colour="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        if colour not in stages_remaining.keys():
                            #print("Stage added for colour ", colour, not_added)
                            stages_known_l[colour] = new_nodes
                            stage_added    = True

                            
                    #stages_remaining.update(stages) = dict(stages.items()+stages_remaining.items())
                    # Add colours to node map
                    # If previously node was another colour, this changes it to the new colour
                    for node in new_nodes:
                        colour_scheme[node] = colour

        # v2 approach from largest context subsets to empty context
        # equivalent to v0 since large context subsets imply small common contexts
        #print("level ", level, "done, number of stages here is now", len(stages_known_l))
        csi_stages.update(stages_known_l)
        #Once we are done with all nodes of current level, move to next level
        level +=1
        #all_stages.append(stages) # list containing stage dict at each level
    #print("Created ", new_stages, "new stages")
    #print("For this tree ", less_data_counter, " assumed p value 0 because of <5 samples")        
    #print("Skipped {} tests".format(skipped))
    #print("Not Skipped {} tests".format(not_skipped)) 
    #print(" Skipped/Not skipped ratio {}".format(skipped/(not_skipped+skipped)))
    #print("COLOURING CSTREE ENDEd WITH STAGES ", stages)
    assert len(csi_stages)<=len(stages)
    
    return c, csi_stages, colour_scheme




def cstree_pc(dataset, 
               val_dict              = None,
               pc_method             = "pgmpy", 
               ordering              = None, 
               draw_cpdag            = True, 
               draw_all_mec_dags     = False,
               draw_mec_dag          = False,
               draw_mc_dags          = False,
               draw_tree_after_dag   = False,
               draw_tree_after_tests = False,
               test                  = "epps", 
               use_dag               = True):
    
    
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
    
    # Generating all DAGs inthe MEC
    dags_bn = []
    all_arcs = cpdag.all_dags()
    for dags in all_arcs:
        temp_graph = nx.DiGraph()
        temp_graph.add_edges_from(dags)
        temp_graph.add_nodes_from([i for i in range(1, dataset.shape[1]+1)])
        dags_bn.append(temp_graph)
        
        if draw_all_mec_dags:
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

    cpdag_edges = len(dags_bn[0].edges)

    
    for mec_dag_num, mec_dag in enumerate(dags_bn):
        if len(mec_dag.edges)!= cpdag_edges:
            continue

        
        if ordering and found_dag:
            orderings=[ordering]
        else:
            orderings=nx.all_topological_sorts(mec_dag)

        if draw_mec_dag:
            fig = plt.figure()
            fig.suptitle("MEC DAG number {}".format(mec_dag_num), fontsize=13)
            nx.draw(mec_dag,pos=nx.drawing.layout.shell_layout(mec_dag),with_labels=True)
            plt.show()

        for order_num, ordering in enumerate(orderings):
            cstree_count+=1

            time_for_wholetree=time.time()

            # Generate CSTree from DAG
            t1=time.time()
            if not use_dag:
                mec_dag = None
            tree,stages,cs = dag_to_cstree(val_dict, ordering, mec_dag)
            #print("Time taken to generate CSTree from MEC DAG {}, ordering number {} is {}".format(mec_dag_num, order_num, time.time()-t1))

            stages_after_dag = len(stages)


            # Perform further context-specific tests
            t1=time.time()
            tree, stages, cs = colour_cstree(tree, ordering, dataset, stages.copy(), cs.copy(), test=test)

            stages_after_csitests = len(stages)

            assert stages_after_dag >= stages_after_csitests
            
            csi_rels_tree = stages_to_csi_rels(stages.copy(), ordering)

            print("from tree we get", csi_rels_tree)

            csi_rels = graphoid_axioms(csi_rels_tree.copy())

            mctemp  = binary_minimal_contexts(csi_rels.copy(), val_dict)

            def equal_dags(g1,g2):
                for n1 in g1.nodes:
                    for n2 in g2.nodes:
                        if (n1 not in list(g2.nodes) or n2 not in list(g1.nodes)):
                            return False
                for e1 in g1.edges:
                    for e2 in g2.edges:
                        if (e1 not in list(g2.edges)) or (e2 not in list(g1.edges)):
                            return False
                return True

            all_mc_graphs = minimal_context_dags(ordering, csi_rels.copy(), val_dict, mec_dag.copy())

            
            for mc,g in all_mc_graphs:
                assert mc == ()
                if mc==():
                    g = nx.relabel_nodes(g,lambda x:int(x))
                    mec_dag = nx.relabel_nodes(mec_dag, lambda x:int(x))
                    
                    print("mec",mec_dag.edges)
                    print("emp",g.edges,"\n")
                    #print("minimalcontexts are", mctemp)
                    assert equal_dags(g, mec_dag)
                


            mecdag_count=0
            if draw_mc_dags:
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


def plot_cstree(tree, colour_scheme):
    pass
