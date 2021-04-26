from utils import cpdag_to_dags, generate_vals, parents,generate_state_space
from cstree import stages_to_csi_rels, dag_to_cstree, color_cstree
from utils import generate_dag, generate_state_space, nodes_per_tree
from graphoid import graphoid_axioms
from mincontexts import minimal_context_dags, binary_minimal_contexts
import networkx as nx
from pgmpy.estimators import PC
from causaldag import pdag
import pandas as pd
import matplotlib.pyplot as plt
from pc import estimate_cpdag, estimate_skeleton

from networkx.drawing.nx_agraph import graphviz_layout

class CSTree(object):
    def __init__(self, dataset, val_dict=None):
        self.dataset=dataset
        if val_dict:
            self.val_dict=val_dict
        else:
            # Assumes each value for each variable occurs
            # atleast once in the dataset
            self.val_dict = generate_state_space(dataset)


    def cpdag_from_pc(self, pc_method="pgmpy"):
        if pc_method not in ["pgmpy","pc"]:
            raise ValueError("PC implementation to use not valid")
        if pc_method == "pgmpy":
            cpdag_model = PC(pd.DataFrame(self.dataset, columns=[i+1 for i in range(self.dataset.shape[1])]))
            cpdag_pgmpy = cpdag_model.estimate(return_type="cpdag")
            cpdag = nx.DiGraph()
            cpdag.add_nodes_from([i+1 for i in range(self.dataset.shape[1])])
            cpdag.add_edges_from(list(cpdag_pgmpy.edges()))
        if pc_method=="pc":
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

        
        return cpdag

    
                                 
    def all_mec_dags(self, pc_method="pgmpy"):
        # Wrapper to get the DAG as a networkx DiGraph
        cpdag = self.cpdag_from_pc(pc_method)
        cpdag = pdag.PDAG(nodes=cpdag.nodes, edges=cpdag.edges)
        dags_bn = []
        all_arcs = cpdag.all_dags()
        for dags in all_arcs:
            temp_graph = nx.DiGraph()
            temp_graph.add_edges_from(dags)
            temp_graph.add_nodes_from([i for i in range(1, self.dataset.shape[1]+1)])
            dags_bn.append(temp_graph)
        return dags_bn


    def cstree_bic(tree, stages):
        pass



    def visualize(self,
                  ordering=None,
                  use_dag=True,
                  all_trees=True,
                  dag=None,
                  pc_method="pgmpy",
                  csi_test="anderson",
                  limit = 5,
                  save_dir=None):
        iteration=0
        trees = self.learn(ordering, use_dag,all_trees, dag, pc_method,csi_test)
        for (tree, stages, color_scheme, ordering) in trees:
            # Save information like CSI relations from it etc
            iteration+=1
            if iteration==limit:
                break

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
            nodes = len(all_mc_dags[0][1].nodes)
            fig = plt.figure(figsize=(24,12))
            main_ax = fig.add_subplot(111)
            tree_ax = plt.subplot(2,1,2)
            dag_ax  = [plt.subplot(2, num_mc_dags, i+1) for i in range(num_mc_dags)]
            tree_node_colors = [color_scheme.get(n, "#FFFFFF") for n in tree.nodes]

            if nodes < 8:
                tree_pos = graphviz_layout(tree, prog="dot", args="")
            else:
                tree_pos = graphviz_layout(tree, prog="twopi", args="")

            nx.draw(tree, node_color=tree_node_colors, ax=tree_ax, pos=tree_pos,
                    with_labels=False, font_color="white", linewidths=1)
            tree_ax.set_title("Ordering is "+"".join(str(ordering)))
            tree_ax.set_ylabel("".join(str(ordering)))
            tree_ax.collections[0].set_edgecolor("#000000")

            for i, (minimal_context, dag) in enumerate(all_mc_dags):
                options = {"node_color":"white", "node_size":1000}
                dag_ax[i].set_title("MC DAG {}".format(minimal_context))
                dag_pos = nx.drawing.layout.shell_layout(dag)
                nx.draw_networkx(dag, pos=dag_pos, ax=dag_ax[i], **options)
                dag_ax[i].collections[0].set_edgecolor("#000000")
            if save_dir:
                plt.savefig(save_dir+str(iteration)+"_cstree_and_mcdags.pdf")
            plt.show()


    

    def learn(self,
              ordering=None,
              use_dag=True,
              all_trees=True,
              dag =None,
              pc_method="pgmpy",
              csi_test="anderson"):

        # If user provides DAG
        if dag:
            dags_bn = [dag]
        # If we start from the CPDAG, generate
        # all DAGs in its MEC
        else:
            dags_bn = self.all_mec_dags(pc_method)

        # To make sure all DAGs above are correct
        mec_dag_edges = len(dags_bn[0].edges)

        # If we are given an ordering, we only want DAGs
        # above which respect this ordering
        if ordering:
            ordering_given=True
            dags_bn = self.all_mec_dags(pc_method)
            dags_bn = [dag for dag in dags_bn if ordering in nx.all_topological_sorts(dag)]
            if dags_bn == []:
                raise ValueError("No DAG with provided ordering in MEC of CPDAG learnt from PC algorithm")

            # we only need one DAG since they all encode the same CSI relations
            dags_bn = [dags_bn[0]]
        else:
            ordering_given=False

        # If we only want to save the best trees
        if not all_trees:
            min_stages = nodes_per_tree(self.val_dict)+1

        # To store the CSTrees and resulting stages
        trees = []

        # For each DAG in the MEC
        for mec_dag_num, mec_dag in enumerate(dags_bn):
            assert len(mec_dag.edges) == mec_dag_edges

            if ordering_given:
                orderings = [ordering]
            else:
                orderings = nx.all_topological_sorts(mec_dag)

            # For each valid causal ordering
            for ordering_num, ordering in enumerate(orderings):
                if not use_dag:
                    mec_dag=None

                # Generate CSTree from DAG
                print("MEC DAG {} ordering {}, applying DAG CI relations".format(mec_dag_num+1, ordering_num+1))
                tree, stages,color_scheme = dag_to_cstree(self.val_dict, ordering, mec_dag)

                stages_after_dag = (len(tree.nodes)-1)-len(color_scheme)+len(stages)

                # Learn CSI relations
                print("Learning CSI relations")
                tree, stages, color_scheme = color_cstree(tree, ordering, self.dataset, stages.copy(), color_scheme.copy(), test=csi_test)

                stages_after_csitests = (len(tree.nodes)-1)-len(color_scheme)+len(stages)

                if use_dag:
                    assert stages_after_dag>=stages_after_csitests
                print("Adding learnt CSTree to list")
                # If we are saving all trees
                if all_trees:
                    trees.append((tree, stages, color_scheme, ordering))
                # If we are saving trees with the least stages
                else:
                    if stages_after_csitests<min_stages:
                        trees.append((tree, stages, color_scheme, ordering))
                    if stages_after_csitests==min_stages:
                        trees.append((tree,stages,color_scheme, ordering))

        return trees      
                
            
        
            
        
    def best_cstrees(self, use_dag=True):
        return
        

    def all_cstrees(self, use_dag=True, pc_method="pgmpy"):
        pass
        
