from nltk.tree import Tree



def get_lca_length(location1, location2):
    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i+=1
    return i

def compute_lca_mean(y_true, y_pred, hierarchy):
    

    total_dist = 0
    count = 0
    for a,b in zip(y_true, y_pred):

        #a,b = row['True Labels'], row['Predicted Labels']
        if a in {'ART', 'OTH', 'NIF', 'LYA'} or b in {'ART', 'OTH', 'NIF', 'LYA'}:
            continue
        if a == b: 
            continue 
        text1 =  a
        text2 =  b
        leaf_values = hierarchy.leaves()
        #leaf_values = classes
        leaf_index1 = leaf_values.index(text1)
        leaf_index2 = leaf_values.index(text2)

        location1 = hierarchy.leaf_treeposition(leaf_index1)
        location2 = hierarchy.leaf_treeposition(leaf_index2)

        #find length of least common ancestor (lca)
        dist = get_lca_length(location1, location2)
        
        
        total_dist += len(location1) - dist
        count+=1
    return total_dist/count

