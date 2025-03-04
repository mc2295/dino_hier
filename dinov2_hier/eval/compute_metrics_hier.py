from collections import defaultdict
import numpy as np
from sklearn.metrics import pairwise_distances, balanced_accuracy_score
from scipy.spatial.distance import euclidean

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd


import torch
import numpy as np
import h5py
from scipy.spatial.distance import pdist, squareform


from scipy.stats import spearmanr, rankdata



from tqdm import tqdm


from scipy.spatial.distance import cdist
from nltk.tree import Tree
from omegaconf import OmegaConf

def process_file(file_name):
    with h5py.File(file_name, "r") as hf:
        # hf.visititems(print)
        features = torch.tensor(hf["features"][:]).tolist()
        label = int(hf["labels"][()])
    return features, label


def extract_embeddings(test_dir):
    test_files = list(Path(test_dir).glob("*.h5"))
    test_features = []
    test_labels = []
    with ThreadPoolExecutor() as executor:
        futures_test = [executor.submit(process_file, file_name) for file_name in test_files]

        for i, future in tqdm(enumerate(futures_test), desc="Loading test data"):
            features, label = future.result()
            test_features.append(features)
            test_labels.append(label)

    test_labels = np.array(test_labels)
    test_features = np.array(test_features)
    return test_labels, test_features


def compute_spearman_ranking(test_labels, test_features, tree_distance):
    unique_labels = np.unique(test_labels)

    avg_distances = np.zeros((len(unique_labels), len(unique_labels)))
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

def get_lca_length(location1, location2):
    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i+=1
    return i

def get_labels_from_lca(ptree, lca_len, location):
    labels = []
    for i in range(lca_len, len(location)):
        labels.append(ptree[location[:i]].label())
    return labels

def get_ancestor_path(label, hierarchy):
    path = set()
    for parent, children in hierarchy.items():
        if label in children and parent != 'root':
            path.add(parent)
            path.update(get_ancestor_path(parent, hierarchy))
    path.add(label)  # Include the label itself
    return path

def hierarchical_metrics(y_true, y_pred, hierarchy, label_to_hier):
    y_true, y_pred = [label_to_hier[i] for i in y_true],[label_to_hier[i] for i in y_pred]

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

def build_parent_map(hierarchy):
    parent_map = {}
    for parent, children in hierarchy.items():
        for child in children:
            parent_map[child] = parent  # Store parent for each child
    return parent_map

def get_top_level_parent(label, parent_map):
    while label in parent_map and parent_map[label] != "Root":
        label = parent_map[label]  # Move up the hierarchy
    return label


def compute_davies_bouldin_index(features, labels):

    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    # Compute centroids for each class
    centroids = {label: np.mean(features[labels == label], axis=0) for label in unique_labels}
    
    # Compute intra-class variance (σ_i)
    intra_class_variance = {
        label: np.mean([euclidean(x, centroids[label]) for x in features[labels == label]])
        for label in unique_labels
    }
    
    # Compute Davies-Bouldin Index
    db_index = 0
    for i, label_i in enumerate(unique_labels):
        max_ratio = float('-inf')
        for j, label_j in enumerate(unique_labels):
            if i == j:
                continue
            # Compute inter-class centroid distance
            inter_class_distance = euclidean(centroids[label_i], centroids[label_j])
            
            # Compute DB ratio for class i with respect to class j
            db_ratio = (intra_class_variance[label_i] + intra_class_variance[label_j]) / inter_class_distance
            max_ratio = max(max_ratio, db_ratio)
        
        db_index += max_ratio
    
    # Average over all classes
    db_index /= num_clusters
    return db_index


def compute_lca_mean(y_true, y_pred, hierarchy, label_to_hier):


    total_dist = 0
    count = 0
    for a,b in zip(y_true, y_pred):
        
        #a,b = row['True Labels'], row['Predicted Labels']
        if a in {'ART', 'OTH', 'NIF', 'LYA'} or b in {'ART', 'OTH', 'NIF', 'LYA'}:
            continue
        if a == b: 
            continue 
        text1 =  label_to_hier[a]
        text2 =  label_to_hier[b]
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
