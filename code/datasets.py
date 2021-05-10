import numpy as np
import networkx as nx
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
import bnlearn
import random

from utils.utils import dag_topo_sort, parents
#from utils.COMBO.combofunc import COMBO
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVR, LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Remember to cite scikit-learn


def dermatology_data(grouped=True):
    # Remember to cite https://archive.ics.uci.edu/ml/datasets/Dermatology
    dermatology_pd = pd.read_csv("../datasets/dermatology.csv")
    print(dermatology_pd.shape)
    dermatology_pd.columns=[
        "1: erythema",
        "2: scaling",
        "3: definite borders",
        "4: itching",
        "5: koebner phenomenon",
        "6: polygonal papules",
        "7: follicular papules",
        "8: oral mucosal involvement",
        "9: knee and elbow involvement",
        "10: scalp involvement",
        "11: family history, (0 or 1)",
        "34: Age (linear)",
        "12: melanin incontinence",
        "13: eosinophils in the infiltrate",
        "14: PNL infiltrate",
        "15: fibrosis of the papillary dermis",
        "16: exocytosis",
        "17: acanthosis",
        "18: hyperkeratosis",
        "19: parakeratosis",
        "20: clubbing of the rete ridges",
        "21: elongation of the rete ridges",
        "22: thinning of the suprapapillary epidermis",
        "23: spongiform pustule",
        "24: munro microabcess",
        "25: focal hypergranulosis",
        "26: disappearance of the granular layer",
        "27: vacuolisation and damage of basal layer",
        "28: spongiosis",
        "29: saw-tooth appearance of retes",
        "30: follicular horn plug",
        "31: perifollicular parakeratosis",
        "32: inflammatory monoluclear inflitrate",
        "33: band-like infiltrate",
        "35: predictor"]
    # -2 because we omit age and predictor in last variable
    X, y = dermatology_pd.values[:,:-2], dermatology_pd.values[:,-1].astype('int')
    estimator = LinearSVC()
    #estimator = SVR(kernel="linear")
    #estimator=LogisticRegressionCV()
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X,y)
    feature_indices = [i for i in range(X.shape[1]) if selector.support_[i]]
    feature_names   = [dermatology_pd.columns[i] for i in feature_indices]
    #+[dermatology_pd.columns[-1]]

    # Select the dataframe with these features
    reduced_dermatology_pd = dermatology_pd[feature_names]

    #sns.pairplot(selected_features, hue=dermatology_pd.columns[-1], diag_kind="hist")
    #plt.show()
    print("Selected features are {}".format(feature_names))
    if "11: family history, (0 or 1)" in feature_names:
        print("Warning family history chosen as a feature, the mapping from scores to binary values need manual fixing")

    if grouped:
        dermatology_dict = {0:0,1:0,2:1,3:1}
        reduced_dermatology_pd = reduced_dermatology_pd.replace(dermatology_dict)
        
    dermatology_np = reduced_dermatology_pd.values.astype(np.int)

    # Put the predictor values in
    dermatology_np  = np.concatenate((dermatology_np, y.reshape(y.shape[0],1)),axis=1).astype(np.int)
    #print(np.unique(dermatology_np[:,-1]))

    return dermatology_np
    

    # Partition each feature into 2 groups which jointly maximize some score
    # Using Combinatorial Bayesian optimization
    # 1 - {1}{2,3,4}, 2 - {1,2}{3,4}, 3 - {1,2,3}{4}


def micecortex_data(num_features=10):
    # Requires xlrd package
    micecortex_pd = pd.read_excel("../datasets/micecortex.xls")

    #for col in micecortex_pd.columns:
    #    print(col, micecortex_pd[col].isna().sum())

    micecortex_pd = micecortex_pd.drop(columns=['BAD_N', 'BCL2_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N',"MouseID", "Genotype", "Treatment",  "Behavior"])

    # TODO Apply reindex
    micecortex_pd = micecortex_pd.dropna()

    # Automatically mapping non-numerical categories to ints
    for col in micecortex_pd.select_dtypes(exclude=["int","double","float" ]).columns:
        micecortex_pd[col] = pd.Categorical(micecortex_pd[col], categories=micecortex_pd[col].unique()).codes
        

    X, y = micecortex_pd.values[:,:-1], micecortex_pd.values[:,-1]
    estimator = LinearSVC()
    #selector = RFECV(estimator, step=1, cv=5)
    selector = RFE(estimator, n_features_to_select=num_features, step=1)
    selector = selector.fit(X,y)
    feature_indices = [i for i in range(X.shape[1]) if selector.support_[i]]
    feature_names   = [micecortex_pd.columns[i] for i in feature_indices]+[micecortex_pd.columns[-1]]

    reduced_micecortex_pd = micecortex_pd[feature_names]

    # We divide each feature into 2 classes based on the median
    reduced_micecortex_medians = reduced_micecortex_pd.median()

    for i,col in enumerate(reduced_micecortex_pd.columns[:-1]):
        reduced_micecortex_pd[col] =  (reduced_micecortex_pd[col]> reduced_micecortex_medians[i]).astype(int)

    micecortex_np = reduced_micecortex_pd.values.astype(np.int)

    #micecortex_np = np.concatenate((micecortex_np, y.reshape(y.shape[0],1)),axis=1).astype(np.int)

    return micecortex_np


def susy_data(laptop, ratio):
    if laptop:
        susy_pd      = pd.read_csv("../../../../Downloads/SUSY.csv")
    else:
        susy_pd =  pd.read_csv("../datasets/SUSY.csv")


    susy_pd      = susy_pd.sample(frac=ratio, replace=True, random_state=1)
    susy_medians = susy_pd.median()

    for i,col in enumerate(susy_pd.columns[1:]):
        susy_pd[col] =  (susy_pd[col]> susy_medians[i]).astype(int)
    susy_np = (susy_pd.values[1:,:9]).astype(np.int)
    print("Dataset has shape {}".format(susy_np.shape))
    return susy_np
    

def coronary_data():
    coronary_pd      = pd.read_csv("../datasets/coronary.csv")
    coronary_pd.columns = ["0:id", "1:S", "2:MW", "3:PWW", "4:P", "5:L", "6:F"]
    coronary_dict      = {"yes":1,"no" :0,"<140":0, ">140":1, "<3":0, ">3":1, "neg":0,"pos":1}
    # Convert outcomes into numerical values
    coronary_pd      = coronary_pd.replace(coronary_dict)
    # Convert into numpy array, 1st row is column names
    # and first column had id so we ignore these
    coronary_np      = coronary_pd.values[:,1:].astype(np.int)
    return coronary_np

def bnlearn_data(dataset_name,n):
    available_sets = ["sprinkler","alarm","andes","asia","pathfinder","sachs","miserables"]

    if dataset_name not in available_sets:
        raise ValueError("Dataset {} not in Python bnlearn".format(dataset_name))

    dag = bnlearn.import_DAG(dataset_name)
    dataset_pd = bnlearn.sampling(dag,n=n)
    dataset_np = dataset_pd.values[1:,:].astype(np.int)
    return dataset_np


def synthetic_dag_binarydata(dag_received, n):
    p = len(dag_received.nodes)
    # Getting nodes with no edges to sample later on
    separate_nodes = []
    for node in dag_received.nodes:
        if list(dag_received.in_edges(node)) ==  list(dag_received.out_edges(node)):
            separate_nodes.append(node)

    # First making the dag only with nodes having edges
    # We sample data from nodes without edges later on
    # bnlearn requires it this way
    dag = nx.DiGraph()
    dag_edges = [e for e in dag_received.edges]
    dag.add_edges_from(dag_edges)
    
    dag = nx.relabel_nodes(dag, lambda x: str(int(x)))
    
    ordering = dag_topo_sort(dag)

    vars_w_no_parents = [n for n in list(dag.nodes) if parents(dag,n)==[]]

    cpds = []


    for var in vars_w_no_parents:
        pr = np.random.rand()
        cpd=TabularCPD(variable=var,
                               variable_card=2,
                               values = [[pr],
                                         [1-pr]])
        print(cpd)
        cpds.append(cpd)

    vars_w_parents = [i for i in ordering if i not in vars_w_no_parents]

    for var in vars_w_parents:
        parents_var = parents(dag,var)

        num_rows = 2**(len(parents_var))

        p_table  = np.zeros((2, num_rows))

        for row in range(num_rows):

            low_p  = np.random.uniform(0.01, 0.2)
            high_p = np.random.uniform(0.8,0.99)

            coin_flip = np.random.rand()
            if coin_flip<0.5:
                pr = low_p
            else:
                pr = high_p

            p_table[0,row] = pr
            p_table[1,row] = 1-pr
        cpd = TabularCPD(variable=var,
                               variable_card=2,
                               values = p_table.tolist(),
                               evidence = parents_var,
                               evidence_card = [2]*len(parents_var))
        print(cpd)
        cpds.append(cpd)

        
    df  = bnlearn.sampling(bnlearn.make_DAG(list(dag.edges), CPD=cpds), n)

        
    dataset = np.zeros((n,p))
        
        # Adding samples from nodes with no edges
    for i in range(p):
        var = i+1
        if var in separate_nodes:
            pr      = np.random.rand()
            samples = np.random.randint(0,2,size=(n,))
        else:
            samples = df[str(var)]
        dataset[:,i] = samples

    return dataset.astype(np.int)
        

        

        
