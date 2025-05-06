from dinov2_hier.eval.utils.compute_lca import compute_lca_mean
from dinov2_hier.loss import load_hierarchy
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, rankdata
from collections import defaultdict
from nltk.tree import Tree


class_mappings = {
    'acevedo': {'basophil': 0, 'eosinophil': 1, 'erythroblast': 2, 'lymphocyte_typical': 3, 'metamyelocyte': 4, 'monocyte': 5, 'myelocyte': 6, 'neutrophil_band': 7, 'neutrophil_segmented': 8, 'promyelocyte': 9},
    'bonemarrow': {'eosinophil_abnormal': 0, 'basophil': 1, 'blast': 2, 'erythroblast': 3, 'eosinophil': 4, 'fagott_cell': 5, 'hairy_cell': 6, 'smudge_cell': 7,  'lymphocyte_typical': 8, 'metamyelocyte': 9, 'monocyte': 10, 'myeloblast': 11, 'neutrophil_band': 12, 'neutrophil_segmented': 13,'proerythroblast': 14, 'plasma_cell': 15, 'promyelocyte': 16},
}
dic_lineage = {
    "lymphoblast": 'lymphopoiesis',
    "myeloblast" : 'blast',
    "blast": "blast",
    "lymphocyte_immature" : 'lymphopoiesis',
    "lymphocyte_reactive": "lymphopoiesis",
    "lymphocyte_typical": "lymphopoiesis",
    "plasma_cell": "lymphopoiesis",
    "hairy_cell": "lymphopoiesis",
    "lymphocyte_large_granular": "lymphopoiesis",
    "lymphocyte_neoplastic": "lymphopoiesis",
    "metamyelocyte": "granulopoiesis",
    "myelocyte": "granulopoiesis",
    "promyelocyte": "granulopoiesis",  
    "neutrophil_band": "granulopoiesis",
    "neutrophil_segmented": "granulopoiesis",
    "promyelocyte_bilobed": "granulopoiesis",
    "eosinophil_abnormal": "granulopoiesis",
    "eosinophil":  "granulopoiesis",
    "basophil":  "granulopoiesis",
    "monoblast": "monocytopoiesis",
    "monocyte": "monocytopoiesis",
    "fagott_cell": "monocytopoiesis",
    "platelet": "platelet",
    "proerythroblast": "erythropoiesis",
    "erythroblast": "erythropoiesis",
    "smudge_cell": "artefact"
}


def compute_spearman_ranking(test_labels, test_features, dataset):
    if dataset == 'bonemarrow':
        tree_distances = np.array([[0., 2., 5., 5., 1., 5., 6., 5., 7., 4., 6., 4., 3., 3., 5., 7., 5.],
            [2., 0., 4., 4., 2., 4., 5., 4., 6., 3., 5., 3., 2., 2., 4., 6., 4.],
            [5., 4., 0., 3., 5., 5., 4., 3., 5., 4., 4., 4., 5., 5., 3., 5., 5.],
            [5., 4., 3., 0., 5., 5., 4., 3., 5., 4., 4., 4., 5., 5., 1., 5., 5.],
            [1., 2., 5., 5., 0., 5., 6., 5., 7., 4., 6., 4., 3., 3., 5., 7., 5.],
            [5., 4., 5., 5., 5., 0., 6., 5., 7., 2., 6., 2., 5., 5., 5., 7., 1.],
            [6., 5., 4., 4., 6., 6., 0., 4., 2., 5., 5., 5., 6., 6., 4., 2., 6.],
            [5., 4., 3., 3., 5., 5., 4., 0., 5., 4., 4., 4., 5., 5., 3., 5., 5.],
            [7., 6., 5., 5., 7., 7., 2., 5., 0., 6., 6., 6., 7., 7., 5., 1., 7.],
            [4., 3., 4., 4., 4., 2., 5., 4., 6., 0., 5., 1., 4., 4., 4., 6., 2.],
            [6., 5., 4., 4., 6., 6., 5., 4., 6., 5., 0., 5., 6., 6., 4., 6., 6.],
            [4., 3., 4., 4., 4., 2., 5., 4., 6., 1., 5., 0., 4., 4., 4., 6., 2.],
            [3., 2., 5., 5., 3., 5., 6., 5., 7., 4., 6., 4., 0., 1., 5., 7., 5.],
            [3., 2., 5., 5., 3., 5., 6., 5., 7., 4., 6., 4., 1., 0., 5., 7., 5.],
            [5., 4., 3., 1., 5., 5., 4., 3., 5., 4., 4., 4., 5., 5., 0., 5., 5.],
            [7., 6., 5., 5., 7., 7., 2., 5., 1., 6., 6., 6., 7., 7., 5., 0., 7.],
            [5., 4., 5., 5., 5., 1., 6., 5., 7., 2., 6., 2., 5., 5., 5., 7., 0.]])

    elif dataset == 'acevedo':
        tree_distances = np.array([[0., 2., 4., 6., 3., 5., 3., 2., 2., 4.],
            [2., 0., 5., 7., 4., 6., 4., 3., 3., 5.],
            [4., 5., 0., 5., 4., 4., 4., 5., 5., 5.],
            [6., 7., 5., 0., 6., 6., 6., 7., 7., 7.],
            [3., 4., 4., 6., 0., 5., 1., 4., 4., 2.],
            [5., 6., 4., 6., 5., 0., 5., 6., 6., 6.],
            [3., 4., 4., 6., 1., 5., 0., 4., 4., 2.],
            [2., 3., 5., 7., 4., 6., 4., 0., 1., 5.],
            [2., 3., 5., 7., 4., 6., 4., 1., 0., 5.],
            [4., 5., 5., 7., 2., 6., 2., 5., 5., 0.]])
       
    unique_labels = np.unique(test_labels)


    avg_distances = np.zeros_like(tree_distances)
    for i, class1 in enumerate(unique_labels):
        for j, class2 in enumerate(unique_labels):
            if i <= j:  # Only compute for upper triangle and diagonal
                features_class1 = test_features[test_labels == class1]
                features_class2 = test_features[test_labels == class2]
                
                # Compute pairwise distances between all vectors of the two classes
                pairwise_distances = cdist(features_class1, features_class2, metric="euclidean")
                
                # Compute average distance
                avg_distances[i, j] = pairwise_distances.mean()
                avg_distances[j, i] = avg_distances[i, j]  # Symmetric matrix

    tree_distances_flat = tree_distances[np.triu_indices_from(tree_distances, k=1)]
    avg_distances_flat = avg_distances[np.triu_indices_from(avg_distances, k=1)]
    correlation, p_value = spearmanr(tree_distances_flat, avg_distances_flat)
    return correlation, p_value

def get_ancestor_path(label, hierarchy):
    path = set()
    for parent, children in hierarchy.items():
        if label in children and parent != 'root':
            path.add(parent)
            path.update(get_ancestor_path(parent, hierarchy))
    path.add(label)  # Include the label itself
    return path

def tree_to_dict(nltk_tree):

    hierarchy_dict = defaultdict(list)

    def traverse_tree(parent, tree):
        if isinstance(tree, Tree):  # If it's a non-leaf node
            hierarchy_dict[parent].append(tree.label())  # Add the node to parent
            for child in tree:
                traverse_tree(tree.label(), child)
        else:  # If it's a leaf node (string label)
            hierarchy_dict[parent].append(tree)

    traverse_tree("root", nltk_tree)  # Start traversal from the root
    return dict(hierarchy_dict)

def hierarchical_metrics(y_true, y_pred, hierarchy):

    num_samples = len(y_true)
    total_intersection = 0
    total_pred_size = 0
    total_true_size = 0

    for i in range(num_samples):
        true_path = get_ancestor_path(y_true[i], hierarchy)
        pred_path = get_ancestor_path(y_pred[i], hierarchy)
        
        intersection = len(true_path & pred_path)
        total_intersection += intersection
        total_pred_size += len(pred_path)
        total_true_size += len(true_path)

    # Compute hierarchical precision and recall
    hP = total_intersection / total_pred_size if total_pred_size > 0 else 0
    hR = total_intersection / total_true_size if total_true_size > 0 else 0

    # Compute hierarchical F1-score
    hF1 = (2 * hP * hR) / (hP + hR) if (hP + hR) > 0 else 0

    return hP, hR, hF1



def compute_hier_metric(y_true, y_pred, y_features, dataset_name):

    dic_mapping = class_mappings[dataset_name] 
    int_to_label = {i:j for j,i in dic_mapping.items()}
    
    hierarchy = load_hierarchy()
    y_true_str = [int_to_label[i] for i in y_true ]
    y_pred_str = [int_to_label[i] for i in y_pred]

    lca = compute_lca_mean(y_true_str, y_pred_str, hierarchy)
    spearman, _ = compute_spearman_ranking(y_true, y_features, dataset_name)
    hierarchy_dic = tree_to_dict(hierarchy)
    hP, hR, hF1 = hierarchical_metrics(y_true_str, y_pred_str, hierarchy_dic)


    true_lineage = [dic_lineage[i] for i in y_true_str]
    predicted_lineage = [dic_lineage[i] for i in y_pred_str]
    lineage_acc = balanced_accuracy_score(true_lineage,  predicted_lineage)

    if dataset_name == 'acevedo':
        rare_classes = {'promyelocyte' }
    elif dataset_name == 'bonemarrow':
        rare_classes = {'abnormal_eosinophil', 'basophil', 'fagott_cell', 'hairy_cell', 'smudge_cell'}
    rare_classes_pred = [y_pred_str[i] for i in range(len(y_pred_str)) if y_true_str[i] in rare_classes]
    rare_classes_true = [i for i in y_true_str if i in rare_classes]
    rare_classes_acc = balanced_accuracy_score(rare_classes_pred, rare_classes_true)

    return lca, spearman, hP, hR, hF1, lineage_acc, rare_classes_acc
