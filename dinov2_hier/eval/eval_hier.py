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

dic_metric = {
    'Model Path' : [],
    'Mistake distance': [],
    'Spearman Ranking' : [],
    'Hierarchical Precision' : [],
    'Hierarchical Recall': [],
    'Hierarchical F1': [],
    'Lineage Accuracy': [],
    'Abnormality Accuracy': [],
    'Rare Classes Accuracy': [],
    'Davis Bouldin': [],
    'Davis Bouldin Lineage' : [],
}

dic_lineage = {
    "lymphoblast": 'blast',
    "myeloblast" : 'blast',
    "blast": "blast",
    "lymphocyte_immature" : 'blast',
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
    "platelet": "thrombopoiesis",
    "proeryhtroblast": "erythropoiesis",
    "erythroblast": "erythropoiesis",
    "smudge_cell": "artefact"
}

for model_path in folder_list:

    bone_marrow_file = model_path + 'bonemarrow/'
    test_dir = bone_marrow_file + os.listdir(bone_marrow_file)[0] + '/test_data'
    test_labels, test_features = extract_embeddings(test_dir)
    sophia = False
    v1 = True

    for pred_file in [model_path + 'knn_eval/knn_eval_labels_and_predictions.csv', model_path + 'log_reg_eval/log_reg_eval_labels_and_predictions.csv' ]:
    #pred_file = 
        dic_mapping = { 'ABE': 0, 'BAS': 1,'BLA' : 2,'EBO': 3, 'EOS': 4,'FGC': 5, 'HAC': 6, 'KSC': 7, 'LYT': 8,'MMZ': 9, 'MON': 10,  'MYB' : 11, 'NGB': 12, 'NGS': 13,'PEB': 14, 'PLM': 15, 'PMO': 16 }
        
        
        #dic_mapping = {'ABE': 0, 'ART': 1, 'BAS': 2, 'BLA': 3, 'EBO': 4, 'EOS': 5, 'FGC': 6, 'HAC': 7, 'KSC': 8, 'LYI': 9, 'LYT': 10, 'MMZ': 11, 'MON': 12, 'MYB': 13, 'NGB': 14, 'NGS': 15, 'NIF': 16, 'OTH': 17, 'PEB': 18, 'PLM': 19, 'PMO': 20}    
        unique_labels = sorted(dic_mapping.keys())
        label_dict = OmegaConf.load('/home/aih/manon.chossegros/from_valentin/dinov2/configs/train/custom_8_hier_4_levels_2.yaml').label_dict 
        label_to_hier = {label: ','.join([str(k) for k in label_dict[label]  if k!= -1]) for label in unique_labels if label not in {'ART', 'OTH', 'NIF', 'LYA'}}
        int_to_label = {i:j for j,i in dic_mapping.items()}

        df = pd.read_csv(pred_file)
        #df['True Labels'] = [int_to_label[i] for i in df['True Labels'].tolist() if int_to_label[i] not in {'ART', 'OTH', 'NIF', 'LYA'}]
        #df['Predicted Labels'] = [int_to_label[i] for i in df['Predicted Labels'].tolist() if int_to_label[i] not in {'ART', 'OTH', 'NIF', 'LYA'}]
        #y_true, y_pred = df['True Labels'].tolist(), df['Predicted Labels'].tolist()
        y_true = [int_to_label[i] for i in df['True Labels'].tolist() if int_to_label[i] not in {'ART', 'OTH', 'NIF', 'LYA'}]
        y_pred = [int_to_label[i] for i in df['Predicted Labels'].tolist() if int_to_label[i] not in {'ART', 'OTH', 'NIF', 'LYA'}]
        
        
        #np.save('test_labels.npy', test_labels) 
        #np.save('test_features.npy', test_features)
        #test_labels, test_features = np.load('test_labels.npy'), np.load('test_features.npy')

        correlation, p_value = compute_spearman_ranking(test_labels, test_features, tree_distances)

        hierarchy = load_hierarchy(4)
        hierarchy_dic = tree_to_dict(hierarchy)

        test_lineages = [dic_lineage[label_to_hier[int_to_label[i]]] for i in test_labels if int_to_label[i]  not in {'ART', 'OTH', 'NIF', 'LYA'}]
        d = {x: i for i, x in enumerate(set(test_lineages))}
        test_lineages = [d[x] for x in test_lineages]
        true_lineage = [dic_lineage[label_to_hier[i]] for i in y_true]
        predicted_lineage = [dic_lineage[label_to_hier[i]] for i in y_pred]
