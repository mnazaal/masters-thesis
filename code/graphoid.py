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