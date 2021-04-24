union        = lambda sets: set.union(*sets)
intersection = lambda sets: set.intersection(*sets)
empty_intersection = lambda sets: True if intersection(sets) == set() else False
set_difference = lambda A,B: A.difference(B)


def decomposition(csi_rels, pairwise=True):
    new_rels = []
    for csi_rel in csi_rels:
        As    = csi_rel[0]
        BuDs  = csi_rel[1]
        Ss    = csi_rel[2]
        Cs    = csi_rel[3]
        if pairwise:
            # Loop over A means we implicitly apply symmetry 
            for B in BuDs:
                for A in As:
                    new_rel = ({A}, {B}, Ss, Cs)
                    C       = set([var for (var,val) in Cs])
                    if new_rel not in csi_rels and new_rel not in new_rels and empty_intersection([{A},{B},Ss,C]):
                        new_rels.append(new_rel)
        else:
            raise NotImplementedError("Implement decomposition for non pairwise case")
            
    return new_rels

def decomposition1(csi_rel, pairwise=True):
    new_rels = []
    (A, B, S, C) = csi_rel

    if len(B)==1:
        # If |B|=1 we cannot drop any element
        return new_rels
    
    if pairwise:
        for b in B:
            new_rels.append((A, {b}, S, C))

    else:
        raise NotImplementedError("Implement decomposition for non pairwise case")
            
    return new_rels

def weak_union1(csi_rel, pairwise=True):
    new_rels = []
    (A, B, S, C) = csi_rel

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


def weak_union(csi_rels, pairwise=True):
    new_rels = []
    for csi_rel in csi_rels:
        As    = csi_rel[0]
        BuDs  = csi_rel[1]
        Ss    = csi_rel[2]
        Cs    = csi_rel[3]
        if pairwise:
            # Loop over A means we implicitly apply symmetry 
            for B in BuDs:
                D1 = set_difference(BuDs, {B})
                for A in As:
                    D2 = set_difference(As, {A})
                    new_rel = ({A}, {B}, union([Ss, D1, D2]), Cs)
                    C       = set([var for (var,val) in Cs])
                    if new_rel not in csi_rels and new_rel not in new_rels and empty_intersection([{A},{B},union([Ss, D1, D2]),C]):
                        new_rels.append(new_rel)
        else:
            raise NotImplementedError("Implement decomposition for non pairwise case")
    return new_rels

def graphoid_axioms(csi_rels):
    #J = csi_rels
    # TODO Removing the while condition
    J = []
    all_axioms_return_empty =  False
    while not all_axioms_return_empty:
        for csi_rel in csi_rels:
            # Add this relation to the closure
            if csi_rel not in J:
                J.append(csi_rel)

            weak_unioned  = weak_union1(csi_rel)
            decomposed    = decomposition1(csi_rel)

            csi_rels += weak_unioned
            csi_rels += decomposed

            # Remove it from list of relations
            # we use to generate others
            # If no such relation left, we are done
            csi_rels.remove(csi_rel)
            if csi_rels == []:
                all_axioms_return_empty = True

    return J
