from utils.utils import cpdag_to_dags, generate_vals, parents,generate_state_space, generate_dag, generate_state_space, nodes_per_tree, shared_contexts,remove_cycles, dag_to_cpdag
from cstree import stages_to_csi_rels, dag_to_cstree, color_cstree
from graphoid import graphoid_axioms
from mincontexts import minimal_context_dags, binary_minimal_contexts
import networkx as nx
import numpy as np
from pgmpy.estimators import PC, ExhaustiveSearch, HillClimbSearch, BicScore, K2Score
from causaldag import pdag
import pandas as pd
import matplotlib.pyplot as plt
from utils.pc import estimate_cpdag, estimate_skeleton
import random
import math

from networkx.drawing.nx_agraph import graphviz_layout

class CSTree(object):
    def __init__(self, dataset, val_dict=None):
        self.dataset=dataset
        self.contingency_table=None # compute only if we need bic
        self.best_cstrees = None
        if val_dict:
            self.val_dict=val_dict
        else:
            # Assumes each value for each variable occurs
            # atleast once in the dataset
            self.val_dict = generate_state_space(dataset)
        for var in list(self.val_dict.keys()):
            if len(np.unique(np.array(self.val_dict[var])))<2:
                raise ValueError("Each variable must take atleast 2 values")


    def learn_cpdag(self, method="pc1"):
        if method == "pc1":
            cpdag_model = PC(pd.DataFrame(self.dataset, columns=[i+1 for i in range(self.dataset.shape[1])]))
            cpdag_pgmpy = cpdag_model.estimate(return_type="cpdag")
            cpdag = nx.DiGraph()
            cpdag.add_nodes_from([i+1 for i in range(self.dataset.shape[1])])
            cpdag.add_edges_from(list(cpdag_pgmpy.edges()))
        if method=="pc2":
            # If the data is binary we do a different test in the PC algorithm
            binary_data = True if all(list(map(lambda f: True if len(f)==2 else False, list(self.val_dict.values())))) else False

        # Set the test to get CPDAG
            if binary_data:
                pc_test = ci_test_bin
            else:
                pc_test = ci_test_dis
            
        # Get CPDAG skeleton
            (g, sep_set) = estimate_skeleton(indep_test_func=pc_test,
                                         data_matrix=self.dataset,
                                         alpha=0.01)

        # Get the CPDAG
            cpdag = estimate_cpdag(skel_graph=g, sep_set=sep_set)
            cpdag = nx.relabel_nodes(cpdag, lambda x: x+1)
            
        if method == "hill":
            cpdag_model=HillClimbSearch(pd.DataFrame(self.dataset, columns=[str(i+1) for i in range(self.dataset.shape[1])]))
            dag_pgmpy = cpdag_model.estimate()
            dag_pgmpy=nx.relabel_nodes(dag_pgmpy, lambda x:int(x))
            dag = nx.DiGraph()
            dag.add_nodes_from([i+1 for i in range(self.dataset.shape[1])])
            dag.add_edges_from(dag_pgmpy.edges)
            cpdag = dag_to_cpdag(dag)
        
        return cpdag

    
                                 
    def all_mec_dags(self, method="pc1"):
        # Wrapper to get the DAG as a networkx DiGraph
        assert method in ["pc1","pc2", "hill", "all"]
        if method[:2]=="pc" or method=="hill":
            cpdag = self.learn_cpdag(method)

                
        if method=="all":
            if self.dataset.shape[1]>5:
                raise ValueError("Too many nodes, max 5")
            # get the dags in the MEC of the DAG with the best score
            data_pd = pd.DataFrame(self.dataset, columns=[str(i+1) for i in range(self.dataset.shape[1])])
            searcher = ExhaustiveSearch(data_pd, scoring_method=K2Score(data_pd))
            max_score=-100000000000
            for score, model in searcher.all_scores():
                if score>max_score:
                    max_score=score
                    dag_pgmpy = model
            print("best dag is", dag_pgmpy.edges)
            dag_pgmpy=nx.relabel_nodes(dag_pgmpy, lambda x:int(x))
            dag = nx.DiGraph()
            dag.add_edges_from(dag_pgmpy.edges)
            cpdag = dag_to_cpdag(dag)
            

        cpdag= remove_cycles(cpdag)
        cpdag.add_nodes_from([i+1 for i in range(self.dataset.shape[1])])
        print("CPDAG from has edges {}".format(list(cpdag.edges)))
        mec_dags = cpdag_to_dags(cpdag)
        dags_bn  = []
        for g in mec_dags:
            dags_bn.append(g)            

        return dags_bn

        
    

    def cstree_bic(self, tree, stages, color_scheme, ordering):
        # Can vectorize this perhaps
        
        # Step 1.Get the contingency table, with indices for variables ordered 1,...,p
        # Note in terms of indices this is 0,...,p-1
        n,p = self.dataset.shape

        if self.contingency_table is None:
            sizes = list(self.val_dict.values())
            u_shape = tuple(len(sizes[i]) for i in range(p))
            self.contingency_table = np.zeros(u_shape)
            for i in range(self.dataset.shape[0]):
                sample=self.dataset[i,:]
                self.contingency_table[tuple(sample)] +=1
                
        u = self.contingency_table
        
        # Step 2. Compute the likelihood

        all_stages   = {}
        stage_counts = {}
        
        ordering_python_index=[o-1 for o in ordering]
        
        for i in range(1,len(ordering)):
            all_stages[i]=[]
            stage_counts[i]=[]
        
        for node in list(tree.nodes)[1:]:
            level = len(node)
            color =  color_scheme.get(node, "#FFFFFF")
            if color=="#FFFFFF":
                common_c      = node
                common_c_vars = [var for (var,val) in common_c]
                #if common_c_vars not in all_stages[level]:
                all_stages[level].append(common_c_vars)
                stage_counts[level].append(tuple(common_c))
            else:
                nodes = stages[color]
                #print("Nodes of {} are {}".format(level,nodes))
                #print("color {} nodes {}".format(color,nodes))
                common_c = shared_contexts(nodes[0],nodes[1])
                for n1 in nodes:
                    common_c = shared_contexts(common_c,n1)
                common_c_vars = [var for (var,val) in common_c]
                #if common_c_vars not in all_stages[level]:
                all_stages[level].append(common_c_vars)
                stage_counts[level].append(tuple(common_c))
            #print("Node {} has common context {} from color {}".format(node, common_c,color))
            

        # i+1 because index i corresponds to variable i+1
        def u_C(C):
            sum_dimensions = [i for i in range(p) if i+1 not in C]
            table=np.sum(u, axis=tuple(sum_dimensions) )
            #print("table is",table, table.shape)
            return table
        
        
        def u_x_C(x,C):
            #print("attempting to extract", x,"under set C", C)
            #assert len(x)==len(C)
            #if C==[]:
            #    return self.dataset.shape[1]
            #else:
            counts =  u_C(C)[tuple(x)]
            #print("counts",counts)
            return counts



        

        # TODO Store stages by their common context and level rather than colors
        def likelihood(x,debug=False):
            # x is ordered 1....p when extracting from dataset
            x=list(x)
            x_ordered = [x[i] for i in ordering_python_index]
            
            pr=1
            
            for level in range(len(ordering)):
                
                # xk-1 in paper, represents the node from root to currnt level
                x_k       = tuple((ordering[i], x_ordered[i]) for i in range(level))
                
                # color of the node, used to extract the common context
                # i should really remove the use of the color 
                color_x_k = color_scheme.get(x_k, "#FFFFFF")
                if color_x_k == "#FFFFFF":
                    context_of_stage_x_k = x_k
                else:
                    nodes_stage_x_k = stages[color_x_k]
                    context_of_stage_x_k = shared_contexts(nodes_stage_x_k[0],nodes_stage_x_k[1])
                    for n2 in nodes_stage_x_k:
                        context_of_stage_x_k = shared_contexts(context_of_stage_x_k,n2)
                # variables of the 
                context_vars = [var for (var,val) in context_of_stage_x_k]

                

                next_var = ordering[level]
                assert next_var not in context_vars
                
                # since the table is ordered with 1...p, we sort C 
                # note these vars range from 1..p but indexing requires 0..p-1 so we take this into account u_C
                CjUk = sorted(context_vars + [next_var])
                Cj   = sorted(context_vars)
                
                # since the table is ordered 1...p we feed it the values in the same order
                x_CjUk = [x[i] for i in range(p) if i+1 in CjUk]
                x_Cj   = [x[i] for i in range(p) if i+1 in Cj]
                

                numerator   = u_x_C(x_CjUk, CjUk)
                denominator = u_x_C(x_Cj,Cj)
                
                
                                
                if level==0:
                    if len(context_vars)==1:
                        assert context_vars[0]==ordering[0]
                        

                pr = pr*(numerator/denominator)
                if np.log(pr)!=np.log(pr):
                    print(CjUk,Cj,x_CjUk,x_Cj, numerator, denominator)
                #assert len(x_CjUk)>0
                #assert len(x_Cj)>0
                
                if debug:
                    print("level {} x is {} when orderd {}, color {}, x_k is {} contxt vars {}, x_k blong to contxt {}".format(level,x,x_ordered,color_x_k,x_k, context_vars,context_of_stage_x_k))
                    print(CjUk,Cj,x_CjUk,x_Cj)
                    print(pr)
                    print("epty c gives u",u_C([]))
                    if Cj==[]:
                        assert denominator==n
            return pr
        #print("stages",stages)    
        #sprint("all-sttages",all_stages)
        #log_mle = sum(likelihood(self.dataset.T))
        #print(likelihood(self.dataset[np.random.randint(0,n),:],debug=True))
        #Checking if they sum to 1 on vitd dataset which is giving nans somewhere ,
        # # but weirdly add to 1 (which is good)when computing the BIC scores for the other 2 cstree configs     
        """
        tl=0
        for a1 in self.val_dict[1]:
            for a2 in self.val_dict[2]:
                for a3 in self.val_dict[3]:
                    for a4 in self.val_dict[4]:
                        for a5 in self.val_dict[5]:
                            tl+=likelihood(np.array([a1,a2,a3,a4,a5]))
                            if math.isnan(tl):
                                print(likelihood([a1,a2,a3,a4,a5]))
                                print([a1,a2,a3,a4,a5])
        
        print("total lik",tl)
        """
        
        log_mle = sum(list(map(lambda i: np.log(likelihood(self.dataset[i,:])), range(n))))
        free_params = sum([len(set(stage_counts[i]))*(len(self.val_dict[i])-1) for i in range(1,p)])
        #print("free params are",free_params)
        #free_params=0
        
        return log_mle-0.5*free_params*np.log(n)
        
        #mle = math.prod(list(map(lambda x:likelihood(x) , list(self.dataset))))
        
            
        

        
        


    def visualize(self,
                  orderings=None,
                  use_dag=True,
                  return_type="minstages",
                  dag=None,
                  plot_mcdags=False,
                  cpdag_method="pc1",
                  csi_test="anderson",
                  kl_threshold=None,
                  learn_limit = 3,
                  last_var=None,
                  plot_limit = 3,
                  save_dir=None):

        # Learn the CSTrees and visualize them
        iteration=0
        trees = self.learn(orderings, use_dag, return_type, dag, cpdag_method,csi_test,kl_threshold,learn_limit,last_var)
        nodes = len(list(self.val_dict.keys()))
        for (tree, stages, color_scheme, ordering) in trees:
            # Save information like CSI relations from it etc
            if iteration==plot_limit:
                break
            iteration+=1

            if plot_mcdags:
                # CSI relations from tree
                print("Tree {} generating CSI rels from tree".format(iteration))
                csi_rels = stages_to_csi_rels(stages.copy(), ordering)

                # Apply weak union, decomposition, specialization iteratively
                # Intersection and contraction afterwards
                print("Applying graphoid axioms")
                csi_rels = graphoid_axioms(csi_rels.copy(), self.val_dict)

                # Get all minimal context DAGs of this CSTree
                print("Generating minimal contexts and minimal context DAGs")
                all_mc_dags = minimal_context_dags(ordering, csi_rels.copy(), self.val_dict)
                num_mc_dags = len(all_mc_dags)
                fig = plt.figure(figsize=(14,12))
                main_ax = fig.add_subplot(111)
                tree_ax = plt.subplot(2,1,2)
                dag_ax  = [plt.subplot(2, num_mc_dags, i+1, aspect='equal') for i in range(num_mc_dags)]
                tree_node_colors = [color_scheme.get(n, "#FFFFFF") for n in tree.nodes]

                if nodes < 11:
                    except_last = [o for o in ordering[:-1]]
                    cstree_ylabel = "".join(["      $X_{}$           ".format(o) for o in except_last[::-1]])
                    tree_pos = graphviz_layout(tree, prog="dot", args="")
                    tree_ax.set_ylabel(cstree_ylabel)
                else:
                    tree_pos = graphviz_layout(tree, prog="twopi", args="")

                nx.draw_networkx(tree, node_color=tree_node_colors, ax=tree_ax, pos=tree_pos,
                        with_labels=False, font_color="white", linewidths=1)

                tree_ax.collections[0].set_edgecolor("#000000")

                for i, (minimal_context, dag) in enumerate(all_mc_dags):
                    options = {"node_color":"white", "node_size":1000}
                    if minimal_context!=():
                        mcdag_title = "".join(["$X_{}={}$  ".format(minimal_context[i][0],minimal_context[i][1]) for i in range(len(minimal_context))])
                    else:
                        mcdag_title = "Empty context"
                    dag_ax[i].set_title(mcdag_title)
                    if list(dag.edges)!=[]:
                        dag_pos = nx.drawing.layout.shell_layout(dag)
                        nx.draw_networkx(dag, pos=dag_pos, ax=dag_ax[i], **options)
                    else:
                        # Darn equal plot size doesnt work with shell layout
                        nx.draw_networkx(dag, ax=dag_ax[i], **options)
                    dag_ax[i].collections[0].set_edgecolor("#000000")
                if save_dir:
                    plt.savefig("figs/"+save_dir+str(iteration)+"_cstree_and_mcdags.pdf")
                else:
                    plt.show()

            else:
                # If we do not plot the minimal context DAGs
                fig = plt.figure(figsize=(24,24))
                tree_ax = fig.add_subplot(111)
                tree_node_colors = [color_scheme.get(n, "#FFFFFF") for n in tree.nodes]

                if nodes < 7:
                    except_last = [o for o in ordering[:-1]]
                    cstree_ylabel = "".join(["$X_{}$        ".format(o) for o in except_last[::-1]])
                    tree_pos = graphviz_layout(tree, prog="dot", args="")
                    tree_ax.set_ylabel(cstree_ylabel)
                else:
                    except_last = [o for o in ordering[:-1]]
                    cstree_ylabel = "".join(["$X_{}$        ".format(o) for o in except_last[::-1]])
                    tree_pos = graphviz_layout(tree, prog="twopi", args="")
                    tree_ax.set_ylabel(cstree_ylabel)

                nx.draw_networkx(tree, node_color=tree_node_colors, ax=tree_ax, pos=tree_pos,
                        with_labels=False, font_color="white", linewidths=1)

                tree_ax.collections[0].set_edgecolor("#000000")

                if save_dir:
                    plt.savefig("figs/"+save_dir+str(iteration)+"_cstree.pdf")
                else:
                    plt.show()


    def count_orderings(self, cpdag_method="pc1", limit=None):
        orderings_count=0
        dags_bn = self.all_mec_dags(cpdag_method)
        for mec_dag in dags_bn:
            orderings = nx.all_topological_sorts(mec_dag)
            
            # orderings is a generator thus might take
            # too much memory to convert to list then count them
            for _ in orderings:
                orderings_count+=1
        return orderings_count
        

    

    def learn(self,
              orderings=None,
              use_dag=True,
              return_type="minstages",
              dag =None,
              cpdag_method="pc1",
              csi_test="anderson",
              kl_threshold=None,
              learn_limit=None,
              last_var=None):
        # Learn the CSTrees and return them as a list containing
        # the tree, its non-singleton stages, ordering and a dictionary with the color scheme

        # If we want to compute the BIC separately, one can always gather
        # the outputs and use the cstree_bic metod on it
        if return_type=="maxbic":
            # A flag for BIC even if we dont use it as the basis
            # for CSTree selection since it can be helpful, but 
            max_no_dag_bic=-1000000000000
            max_dag_bic  = -1000000000000
            max_pc_bic   = -1000000000000
            max_bic      = -1000000000000
        elif return_type=="minstages":
            # We need a way to store BIC since there could be many trees with the same
            # same stage in which case we want to store the highest BIC
            min_stages            =  1000000000000
            min_dag_stages        =  1000000000000
            min_no_dag_stages     =  1000000000000
            min_pc_stages         =  1000000000000
            min_dag_stages_bic    = -1000000000000
            min_no_dag_stages_bic = -1000000000000
            min_pc_stages_bic     = -1000000000000
        else:
            if return_type!="all":
                raise ValueError("Return type must be maxbix,minstages,or all, not {}".format(return_type))
        
        

        # If user provides DAG
        if dag:
            dags_bn = [dag]
        # If we start from the CPDAG, generate
        # all DAGs in its MEC
        else:
            dags_bn = self.all_mec_dags(cpdag_method)
            if len(dags_bn)==0:
                raise ValueError("No MEC DAGs, check output of PC/hillclimb etc")

        # To make sure all DAGs above are correct by checking if edges are the same
        # for all DAGs in the MEC
        mec_dag_edges = len(dags_bn[0].edges)

        # If we give some orderings, we only want DAGs
        # above which respect this ordering
        if orderings:
            orderings_given = True
            #consistent_dags=[]
            #for o in orderings:
            #    consistent_dags += [dag for dag in dags_bn if o in nx.all_topological_sorts(dag)]
            #dags_bn = consistent_dags
            #if dags_bn == [] and use_dag:
            #    dags_bn=[None]
                #raise ValueError("No DAG with provided ordering in MEC of CPDAG learnt from PC algorithm")
        else:
            orderings_given=False


                # If we have ordering and want a DAG to encode CI relations
            # we only need one DAG since they all encode the same CI relations
            #if use_dag:
            #    dags_bn = [dags_bn_w_ordering[0]]
            #else:

            
        # To store the CSTrees and resulting stages
        trees = []

        # Limits to see results quicker
        #if learn_limit:
        #    if len(dags_bn)>learn_limit:
        #        dags_bn = random.sample(dags_bn,learn_limit)
                
        print("Experiment has {} total orderings".format(self.count_orderings(cpdag_method=cpdag_method)))


        # For each DAG in the MEC
        skipped_orderings=0
        for _, mec_dag in enumerate(dags_bn):
            if use_dag and mec_dag is not None:
                assert len(mec_dag.edges) == mec_dag_edges

            #if ordering_given:
            #    orderings = [ordering]
            if not orderings_given and mec_dag is not None:
                orderings = nx.all_topological_sorts(mec_dag)
                # We need a separate generator to count ordering
                # if we do not want to use it up in the counting process
                #orderings_for_counting =  nx.all_topological_sorts(mec_dag)
                # TODO Put continue statement instead of converting
                # orderings to list which can take long time for some DAGs
                #if learn_limit:
                    # make it work for non-generators
                #    orderings = [next(orderings)]

            #("MEC DAG {} with {} edges which are {} has {} orderings".format(mec_dag_num+1, len(list(mec_dag.edges)), list(mec_dag.edges), "not printed"))


            # For each valid causal ordering
            # Remember orderings is a generator so copy it when you use it
            # since any use of it will empty its elements
            
            for _, ordering in enumerate(orderings):
                
                if orderings_given and ordering not in  nx.all_topological_sorts(mec_dag):
                    print("Current DAG with edges {} not consistent with current given ordering {}".format(mec_dag.edges, ordering))
                    skipped_orderings+=1
                    print(skipped_orderings)
                    temp_dag=None
                else:
                    temp_dag=mec_dag.copy()

                # If the user knows the last variable, we skip orderings
                # where the last variable does not match
                if last_var:
                    if ordering[-1]!=last_var:
                        print("Skipping ordering {} since the last variable is not {}".format(ordering, last_var))
                        continue

                
                # Generate CSTree from DAG
                #print("="*40)
                #print("MEC DAG number {} is {} ordering number {} applying DAG CI relations".format(mec_dag_num+1,mec_dag.edges, ordering_num+1))
                

                # Tree straight from DAG
                #print("DAG to CSTree start")
                tree, stages1 ,color_scheme1,stage_list1,color_scheme_list1 = dag_to_cstree(self.val_dict, ordering, temp_dag, use_dag=True)
                #print("DAG to CSTree end")

                # Tree without DAG CI relations
                #print("CSI alone start")
                tree, stages2, color_scheme2 = color_cstree(tree, ordering,self.dataset, self.val_dict,test=csi_test, kl_threshold=kl_threshold)
                #print("CSI alone end")

                # Tree using DAG CI relations
                #print("CSI w DAG start")
                tree, stages3, color_scheme3 = color_cstree(tree, ordering, self.dataset, self.val_dict, stage_list1.copy(), color_scheme_list1.copy(), test=csi_test, kl_threshold=kl_threshold)
                #print("CSI w DAG end")

                total_tree_nodes = nodes_per_tree(self.val_dict, ordering)
                dag_stages    = total_tree_nodes-len(color_scheme1)+len(stages1)
                no_dag_stages = total_tree_nodes-len(color_scheme2)+len(stages2)

                pc_stages     = total_tree_nodes-len(color_scheme3)+len(stages3)
                dag_bic    = self.cstree_bic(tree, stages1.copy(), color_scheme1.copy(),ordering)
                no_dag_bic = self.cstree_bic(tree, stages2.copy(), color_scheme2.copy(),ordering)
                pc_bic     = self.cstree_bic(tree, stages3.copy(), color_scheme3.copy(),ordering)

                    

                
                #print("CSTree from DAG has {} stages, {} non-singleton stages".format(dag_stages, len(stages1)))
                #print("CSTree without DAG CI relations has {} stages, {} non-singleton stages".format(no_dag_stages, len(stages2)))
                #print("CSTree with DAG CI relations has {} stages, {} non-singleton stages".format(pc_stages, len(stages3)))

                if return_type=="minstages":
                    if dag_stages < min_dag_stages:
                        min_dag_stages = dag_stages
                        min_dag_stages_bic = dag_bic
                    if dag_stages==min_dag_stages:
                        if dag_bic > min_dag_stages_bic:
                            min_dag_stages_bic = dag_bic
                    if no_dag_stages < min_no_dag_stages:
                        min_no_dag_stages = no_dag_stages
                        min_no_dag_stages_bic = no_dag_bic
                    if no_dag_stages == min_no_dag_stages:
                        if no_dag_bic > min_no_dag_stages_bic:
                            min_no_dag_stages_bic = no_dag_bic
                            #print("!!!!!!!!!",no_dag_bic, ordering) #for experiment section
                    if pc_stages < min_pc_stages:
                        min_pc_stages = pc_stages
                        min_pc_stages_bic=pc_bic
                    if pc_stages==min_pc_stages:
                        if pc_bic > min_pc_stages_bic:
                            min_pc_stages_bic=pc_bic
                        


                    
                    min_stages_current_ordering = min(dag_stages, no_dag_stages,pc_stages)
                    if min_stages_current_ordering<min_stages:
                        min_stages=min_stages_current_ordering
                        best_ordering = ordering
                        if min_stages_current_ordering==dag_stages:
                            trees = [(tree, stages1, color_scheme1, ordering)]
                        if min_stages_current_ordering==no_dag_stages:
                            trees = [(tree, stages2, color_scheme2, ordering)]
                        if min_stages_current_ordering==pc_stages:
                            trees = [(tree, stages3, color_scheme3, ordering)]
                    if min_stages_current_ordering==min_stages:
                        if min_stages_current_ordering==dag_stages:
                            trees.append((tree, stages1, color_scheme1, ordering))
                        if min_stages_current_ordering==no_dag_stages:
                            trees.append((tree, stages2, color_scheme2, ordering))
                        if min_stages_current_ordering==pc_stages:
                            trees.append((tree, stages3, color_scheme3, ordering))
                elif return_type=="maxbic":



                    #print("BIC Scores for ordering {} are".format(ordering))
                    #print("Just DAG {}, Without DAG CI relations {}, With DAG CI relations {}".format(dag_bic, no_dag_bic, pc_bic))
                    
                    if dag_bic>max_dag_bic:
                        max_dag_bic           = dag_bic
                        max_dag_bic_stages    = dag_stages
                    if no_dag_bic>max_no_dag_bic:
                        max_no_dag_bic        = no_dag_bic
                        max_no_dag_bic_stages = no_dag_stages
                    if pc_bic>max_pc_bic:
                        max_pc_bic            = pc_bic
                        max_pc_bic_stages     = pc_stages
                    max_bic_current_ordering  = max(dag_bic, no_dag_bic, pc_bic)
                    if max_bic_current_ordering>max_bic:
                        max_bic=max_bic_current_ordering
                        best_ordering = ordering
                        if max_bic_current_ordering == dag_bic:
                            trees = [(tree,stages1,color_scheme1,ordering)]
                        if max_bic_current_ordering == no_dag_bic:
                            trees = [(tree, stages2, color_scheme2, ordering)]
                        if max_bic_current_ordering == pc_bic:
                            trees = [(tree, stages3, color_scheme3, ordering)]
                elif return_type=="all":
                    # In this case we only save the tree as per algorithm 6,
                    # i.e. tree with DAG CI relations and further CSI relations
                    trees.append((tree, stages1, color_scheme1, ordering))
                    trees.append((tree, stages2, color_scheme2, ordering))
                    trees.append((tree, stages3, color_scheme3, ordering))

                
                if learn_limit and len(trees)==learn_limit:
                    #and return_type=="all":
                    break
            if  learn_limit and len(trees)==learn_limit:
                # and return_type=="all":
                break

        if return_type=="minstages":
            print("{} CSTree with same number of minimum stages which is {}".format(len(trees), min_stages))
            print("Min stages: Just DAG {} w BIC {}, Without DAG CI relations {} w BIC {}, With DAG CI relations {} w BIC {}, Min stage ordering {}".format(min_dag_stages, min_dag_stages_bic, min_no_dag_stages, min_no_dag_stages_bic, min_pc_stages, min_pc_stages_bic, best_ordering))

            #self.best_cstrees = trees.copy()
        #if get_bic:
        #    with open("bic_dag.json", 'w') as f:
            # indent=2 is not needed but makes the file human-readable
        #    json.dump(score, f, indent=2)
        elif return_type=="maxbic":
            print("Max BIC Scores : Just DAG {} w stages {}, Without DAG CI relations {} w stages {}, With DAG CI relations {} w stages {}, Max BIC ordering {}".format(max_dag_bic, max_dag_bic_stages, max_no_dag_bic, max_no_dag_bic_stages, max_pc_bic, max_pc_bic_stages, best_ordering))
        return trees      
                
            
        
            
        
    def best_cstrees(self, use_dag=True):
        return
        

    def all_cstrees(self, use_dag=True, cpdag_method="pgmpy"):
        pass
        
