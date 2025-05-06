from torch.utils.data import DataLoader
from dinov2_hier.eval.utils.feature_extraction import save_features_and_labels, get_data
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

acevedo_path = 'path/to/acevedo/images'
bm_path_train = "path/to/bm_train.csv"
bm_path_test = "path/to/bm_test.csv"



model_list = [
    'path/to/model/checkpoints'

]


class_mappings = {
    'ace': {'basophil': 0, 'eosinophil': 1, 'erythroblast': 2, 'lymphocyte_typical': 3, 'metamyelocyte': 4, 'monocyte': 5, 'myelocyte': 6, 'neutrophil_band': 7, 'neutrophil_segmented': 8, 'promyelocyte': 9},
    'bm': {'eosinophil_abnormal': 0, 'basophil': 1, 'blast': 2, 'erythroblast': 3, 'eosinophil': 4, 'fagott_cell': 5, 'hairy_cell': 6, 'smudge_cell': 7,  'lymphocyte_typical': 8, 'metamyelocyte': 9, 'monocyte': 10, 'myeloblast': 11, 'neutrophil_band': 12, 'neutrophil_segmented': 13,'proerythroblast': 14, 'plasma_cell': 15, 'promyelocyte': 16},
}
inv_class_mappings = {
    dataset: {v: k for k, v in mapping.items()}
    for dataset, mapping in class_mappings.items()
}

for source in ['ace', 'bm']:
    for model_path in model_list:
        
        result_dir = 'results/'+ Path(model_path).parent.parent.stem
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        sorted_paths = list(Path(model_path).rglob("*teacher_checkpoint.pth"))
        print(model_path, sorted_paths)


        checkpoint = sorted_paths[0]

        parent_dir=checkpoint.parent 

        print("loading checkpoint: ", checkpoint)

        ### Acevedo
        if source == "ace":

            feature_dir = parent_dir / 'acevedo_eval' 


        if source == 'bm':
            
            feature_dir_bm = parent_dir / 'bm_eval' 
            feature_dir = Path(os.path.join(feature_dir_bm, "test_data"))


        all_features=list(feature_dir.glob("*.h5"))
        data,labels,filenames = get_data(all_features)

        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(data)


        label_names = [inv_class_mappings[source][label] for label in labels]

        # Plotting
        figure = plt.figure(figsize=(10, 8))
        palette = sns.color_palette("tab10", n_colors=len(inv_class_mappings[source]))
        sns.scatterplot(
            x=embedding[:, 0], y=embedding[:, 1],
            hue=label_names,
            palette=palette,
            s=5,
            alpha=0.8,
            edgecolor='k'
        )
        plt.title("UMAP Projection of Cell Types", fontsize=14)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        figure.savefig(result_dir + '/' + source + '/UMAP.png', bbox_inches='tight')