from itertools import combinations
from utils.utils import generate_vals

union        = lambda sets: set.union(*sets)
intersection = lambda sets: set.intersection(*sets)
empty_intersection = lambda sets: True if intersection(sets) == set() else False
set_difference = lambda A,B: A.difference(B)

def decomposition(csi_rel, pairwise=True):
    new_rels = []
    (A, B, S, C) = csi_rel
    #assert set.intersection(A,B,S,set([var for (var,val) in C]))==set()

    if len(B)==1:
        # If |B|=1 we cannot drop any element
        return new_rels
    
    if pairwise:
        for b in B:
            new_rels.append((A, {b}, S, C))

    else:
        raise NotImplementedError("Implement decomposition for non pairwise case")
            
    return new_rels

def specialization(csi_rel, val_dict, pairwise=True):
    new_rels = []
    (A,B,S,C) = csi_rel
    #assert set.intersection(A,B,S,set([var for (var,val) in C]))==set()
    
    if pairwise:
        if len(A)>1:
            raise ValueError("Using pairwise case but |A|>1")
        #C_vars = [var for (var,val) in C]
        for T_size in range(1,len(S)):
            # for each subset size except the empty and full set
            Ts = [i for i in combinations(list(S), T_size)]
            for T in Ts:
                vals_T_takes = generate_vals(list(T), val_dict)
                for x_T in vals_T_takes: 
                    Ci  = C.copy()
                    X_T = [(T[i],x_T[i]) for i in range(len(T))]
                    Ci += x_T
                    new_rels.append((A,B,set(S).difference(T),Ci))
    else:
        raise NotImplementedError("Implement specialization for non pairwise case")
    return new_rels
                


def weak_union(csi_rel, pairwise=True):
    new_rels = []
    (A, B, S, C) = csi_rel
    #assert set.intersection(A,B,S,set([var for (var,val) in C]))==set()

    if len(B)==1:
        # Cannot push any element in B into
        # the conditioning set
        return new_rels

    if pairwise:
        if len(A)>1:
            raise ValueError("Using pairwise case but |A|>1")
        for b in B:
            new_rels.append((A, {b}, S.union(B.difference({b})), C))
    else:
        raise NotImplementedError("Implement weak union for non pairwise case")

    return new_rels

def intersection(csi_rels, memo, pairwise=True):
    new_rels = []
    to_look  = [i for i in csi_rels if i not in memo]
    if pairwise:
        for (rel1, rel2) in combinations(to_look,2):
            (A1,B1,S1,C1) = rel1
            (A2,B2,S2,C2) = rel2
            
            #assert set.intersection(A1,B1,S1,set([var for (var,val) in C1]))==set()
            #assert set.intersection(A2,B2,S2,set([var for (var,val) in C2]))==set()
            if len(A1)>1 or len(A2)>1:
                raise ValueError("Using pairwise case but |A|>1")
            if A1==A2 and set(C1)==set(C2) and B1.union(S1)==B2.union(S2):
                D = S1.intersection(B2)
                if D == set():
                    to_union = S1
                else:
                    to_union = S1.difference(D)
                    
                new_rel = (A1, B1.union(to_union), D, C1)
                if new_rel not in csi_rels and new_rel not in new_rels:
                    new_rels.append(new_rel)
    else:
        raise NotImplementedError("Implement intersection for non pairwise case")
    return new_rels, to_look



def contraction(csi_rels, memo, pairwise=True):
    new_rels = []
    to_look  = [i for i in csi_rels if i not in memo]
    if pairwise:
        for (rel1, rel2) in combinations(to_look,2):
            (A1,B1,S1,C1) = rel1
            (A2,B2,S2,C2) = rel2
            
            #assert set.intersection(A1,B1,S1,set([var for (var,val) in C1]))==set()
            #assert set.intersection(A2,B2,S2,set([var for (var,val) in C2]))==set()
            if len(A1)>1 or len(A2)>1:
                raise ValueError("Using pairwise case but |A|>1")
            if A1==A2 and set(C1)==set(C2) and S1==B2.union(S2):
                new_rel = (A1, B1.union(B2), S2, C1)
                if new_rel not in csi_rels and new_rel not in new_rels:
                    new_rels.append(new_rel)
    else:
        raise NotImplementedError("Implement intersection for non pairwise case")
    return new_rels, to_look           


def graphoid_axioms(csi_rels, val_dict):
    # TODO Removing the while condition
    print("Started applying weak union and decomposition to {} relations".format(len(csi_rels)))
    J = []
    if csi_rels == []:
        all_axioms_return_empty =  True
    else:
         all_axioms_return_empty =  False
    #print("started with {} relations".format(len(csi_rels)))

    intersec_memo = []
    contrac_memo  = []
         
    while not all_axioms_return_empty:
        for csi_rel in csi_rels:
            # Add this relation to the closure
            if csi_rel not in J:
                J.append(csi_rel)
            if len(J)==20:
                print(len(J))

            weak_unioned  = weak_union(csi_rel)
            decomposed    = decomposition(csi_rel)
            #specialized   = specialization(csi_rel, val_dict)
            #intersected, intersec_memo  = intersection(csi_rels.copy()+J.copy(), intersec_memo)
            # contracted, contrac_memo = contraction(csi_rels.copy()+J.copy(), contrac_memo)
            
            csi_rels += weak_unioned
            csi_rels += decomposed
            #csi_rels += specialized
            #csi_rels += intersected
            #csi_rels += contracted


            # Remove it from list of relations
            # we use to generate others
            # If no such relation left, we are done
            csi_rels.remove(csi_rel)
            if csi_rels == []:
                all_axioms_return_empty = True

    print("Applying intersection to {} relations".format(len(J)))
    intersected, _ = intersection(J.copy(), [])
    #contracted,  _ = contraction(J.copy(),[])
    J += intersected
    #J += contracted
    print("Applying specialization to {} relations".format(len(J)))
    for rel in J:
        J+=specialization(rel, val_dict)
    print("Giving {} relations to get minimal contexts".format(len(J)))
    return J
