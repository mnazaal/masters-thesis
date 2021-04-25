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
from utils import nodes_per_tree


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
            # TODO Delete below
            # π_k+1
            next_var       = ordering[level]
 
            # Parents_G(π_k+1)
            pars           = parents(dag, next_var)

            if len(pars)==level:
                # All variables
                # prior to var in
                # ordering are parents
                # thus this level is all white
                continue
            

            preceding_vars   = ordering[:level]
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
    

                else:
                    stage_nodes = [n for n in roots if set(sc).issubset(n)]

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

    if stages == {}:
        use_dag = False
    else:
        use_dag = True

    
    if colour_scheme is None:
        colour_scheme = {}
        coloured = []
    else:
        coloured = list(colour_scheme.keys())
        
    less_data_counter = 0
    num_empty_contexts = 0
    
    levels = len(ordering)
    
    csi_stages = {}

    csi_stages = []

    #print("COLOURING CSTREE STARTED WITH STAGES ", stages)
    
    while level<levels+1:
        stages_removed_l=0
        stages_added_l=0
        # at each level we have a list of lists representing nodes in the same stage        
        
        # Nodes in this level
        nodes_l  = [n for n in c.nodes if nx.shortest_path_length(c, "Root", n)==level]
        print("len of each node in level ",level,list(map(lambda x: len(x), nodes_l)))
        stages_l = {c:ns for c,ns in stages.items() if len(ns[0])==level}
        #stages_l = stages_l.copy()

        #print("starting level {}, nodes len and stages are {},{}".format(level, len(nodes_l), stages_l))
        
        if len(stages_l) == 1:
            level+=1
            csi_stages.update(stages_l)
            continue
        # v0, generate common contexts starting from the empty context to the full contexts
        
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

            
            coloured = list(colour_scheme.keys())
                
            # Case (a) : both coloured and same colour
            #print(set(colour_scheme[n1]))
            if (n1 in coloured and n2 in coloured) and set(colour_scheme.get(n1)) == set(colour_scheme.get(n2)):
                skipped += 1
                #continue
                    
            # Case (b,c,d,e) : Other than above
            else:
                # We do not skip the test if 2 nodes do not have the same colour
                not_skipped+=1
                colour_n1 = colour_scheme.get(n1, None)
                colour_n2 = colour_scheme.get(n2, None)
               # print(level,n1,n2)
                # Case (b)
                if colour_n1 is not None and colour_n2 is not None:
                    common_c_n1 = shared_contexts(stages_l[colour_n1][0][:],
                                                  stages_l[colour_n1][1][:])
                    common_c_n2 = shared_contexts(stages_l[colour_n2][0][:],
                                                  stages_l[colour_n2][1][:])
                    print(stages_l[colour_n2][1])
                    common_c    = shared_contexts(common_c_n1,common_c_n2)
                # Case (c)
                if colour_n1 is not None and colour_n2 is None:
                    common_c_n1 =  shared_contexts(stages_l[colour_n1][0][:],
                                                   stages_l[colour_n1][1][:])
                    common_c    = shared_contexts(common_c_n1, n2[:])
                # Case (d)
                if colour_n1 is None and colour_n2 is not None:
                     common_c_n2 = shared_contexts(stages_l[colour_n2][0][:],
                                                   stages_l[colour_n2][1][:])
                     common_c   = shared_contexts(n1[:], common_c_n2)
                # Case (e)
                if colour_n1 is None and colour_n2 is None:
                    common_c = shared_contexts(n1[:],n2[:])
                
                    

                #common_c = shared_contexts(n1[:-1],n2[:-1])

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
                    stages_added_l+=1
                    #print("adding at level ", level, "commonc", common_c)
                    
                    if common_c == []:
                        num_empty_contexts+=1
                    # Nodes below belong to the same stage defined by the common context

                    new_nodes = [n for n in nodes_l
                                 if set(common_c).issubset(set(n[:]))]
                   
                    # Remove stages where nodes have common context with new stage
                    if colour_n1 is not None:
                        stages_removed_l+=1
                        print("level ",level,"removing nodes with common c", common_c_n1) 
                        keys_to_keep = set(stages_l.keys())-{tuple(common_c_n1)}
                        stages_l = {k:stages_l[k] for k in keys_to_keep}
                        #stages_l = {c:ns for c,ns in stages_l.items()
                        #            if not set(common_c_n1).issubset(set(ns[0][:-1]))}
                    if colour_n2 is not None:
                        stages_removed_l+=1
                        print("level ",level,"removing nodes with common c", common_c_n2) 
                        keys_to_keep = set(stages_l.keys())-{tuple(common_c_n2)}
                        stages_l = {k:stages_l[k] for k in keys_to_keep}
                    

                    # Then add the new stage
                    stage_added = False
                    while not stage_added:
                        # Just to make sure they are different colours
                        color="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        if color not in stages_l.keys():
                            # Add new stage
                            stages_l[tuple(common_c)] = new_nodes
                            stage_added    = True
                    
                    #stages_l.update(stages)
                    # Add colours to node map
                    # If previously node was another colour, change it to the new colour
                    for node in new_nodes:
                        colour_scheme[node] = tuple(common_c)
                    
        csi_stages.append(stages_l)
        #Once we are done with all nodes of current level, move to next level
        level +=1
        print("level ",level,"must have {} stages".format(stages_added_l-stages_removed_l))
        
    if use_dag:
        assert len(csi_stages)<=len(stages)
    print("skip ratio",skipped/(skipped+not_skipped))
    csi_stages1, colour_scheme1 = {},{}
    for stages_per_level in csi_stages:
        for context,nodes in stages_per_level.items():
            color="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            csi_stages1[color] = nodes
            for node in nodes:
                colour_scheme1[node]=color
    
    return c, csi_stages1, colour_scheme1




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

    print(n,p)
    
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


    cstree_best = []
    stages_best = nodes_per_tree(val_dict)-1
    non_empty_mcdags = []
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

            # Generate CSTree from DAG
            if not use_dag:
                mec_dag = None
            tree,stages,cs = dag_to_cstree(val_dict, ordering, mec_dag)
            #print("stages after generating tree")
            #print(stages, "\n")

            stages_after_dag = len(stages)

            # Perform further context-specific tests
            t1=time.time()
            tree, stages, cs = colour_cstree(tree, ordering, dataset, stages.copy(), cs.copy(), test=test)

            #print("stages after csi tests")
            #print(stages , "\n")

            stages_after_csitests = len(stages)

            stages_current = (len(tree.nodes)-1) - (len(cs)) + len(stages)

            if stages_current<stages_best:
                stages_best = stages_current
                cstree_best = [(tree,stages,cs,ordering, mec_dag)]
            if stages_current==stages_best:
                cstree_best.append((tree,stages,cs,ordering, mec_dag))
                
            # TODO this is temp
            if use_dag:
                assert stages_after_dag >= stages_after_csitests
            
            csi_rels_tree = stages_to_csi_rels(stages.copy(), ordering)

            #print("from tree we get", csi_rels_tree)

            csi_rels = graphoid_axioms(csi_rels_tree.copy())

            mctemp  = binary_minimal_contexts(csi_rels.copy(), val_dict)

            #print("order ", ordering)
            #for k,v in mctemp.items():
            #    print("min context", k, "ci rels", v,"\n")


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
            
            if mec_dag is None and use_dag:
                mec_dag = cpdag.copy()

            all_mc_graphs = minimal_context_dags(ordering, csi_rels.copy(), val_dict, mec_dag, csi_rels_tree.copy())

            if len(all_mc_graphs)>1:
                non_empty_mcdags.append((tree.copy(), stages.copy(), cs.copy(), ordering.copy(), cpdag.copy()))


                #f
                nodes = dataset.shape[1]
                case = non_empty_mcdags[0]

                (tree, stages, cs, ordering, mec_dag) = case

                print("stages are\n")
                for c,ns in stages.items():
                    print(c, len(ns) ,len(ns[0])) 
                csi_rels_from_tree = stages_to_csi_rels(stages.copy(),ordering)

                #print("from tree\n", csi_rels_from_tree)

                csi_rels = graphoid_axioms(csi_rels_from_tree.copy())

                #print("after axioms\n", csi_rels)

                minimal_contexts = binary_minimal_contexts(csi_rels.copy(), val_dict)

                #print("minimal contexts\n", minimal_contexts)

                #fig, ax = plt.subplots(2,num_mc_graphs)

                all_mc_graphs = minimal_context_dags(ordering, csi_rels.copy(), val_dict)
                num_mc_graphs = len(all_mc_graphs)


                fig=plt.figure(figsize=(24, 12))
                main_ax = fig.add_subplot(111)
                tree_ax = plt.subplot(2,1,2)
                ax = [plt.subplot(2, num_mc_graphs, i+1) for i in range(num_mc_graphs)]
                node_colors = [cs.get(n, "#FFFFFF") for n in tree.nodes()]
                if nodes<7:
                    pos = graphviz_layout(tree, prog="dot", args="")
                else:
                    pos = graphviz_layout(tree, prog="twopi", args="")
                nx.draw(tree, node_color=node_colors, ax=tree_ax,pos=pos, with_labels=False, font_color="white", linewidths=1)
                tree_ax.set_title("ordering is "+"".join(str(ordering)))
                tree_ax.set_ylabel("".join(str(ordering)))
                tree_ax.collections[0].set_edgecolor("#000000")
                #plt.show()



                for i, (mc,g) in enumerate(all_mc_graphs):
                    options = {"node_color":"white","node_size":1000}
                    ax[i].set_title("MC DAG Context {}".format(mc))
                    nx.draw_networkx(g,pos = nx.drawing.layout.shell_layout(g), ax=ax[i],**options)
                    ax[i].collections[0].set_edgecolor("#000000")

                plt.show()
                return

            #f

            
            for mc,g in all_mc_graphs:
                #assert mc == ()
                if mc==() and use_dag:
                    g = nx.relabel_nodes(g,lambda x:int(x))
                    mec_dag = nx.relabel_nodes(mec_dag, lambda x:int(x))
                    #print(ordering)
                    #print("\ncsi rels tree", csi_rels_tree)
                    #print("mec",mec_dag.edges)
                    #print("emp",g.edges,"\n")
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



                

    return cstree_best, non_empty_mcdags

def plot_cstree(tree, colour_scheme):
    pass
