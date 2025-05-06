"""
taken from feature_extraction branch from HistoBistro
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os
from pathlib import Path
#from utils import create_label_mapping_from_paths

from dinov2_hier.eval.utils.utils import create_label_mapping, create_label_mapping_from_paths


# Acevedo
class PathImageDataset(Dataset):
    def __init__(self, image_path, transform,class_to_label=None,filetype=".tiff",img_size=(224,224)):
        self.images = list(Path(image_path).rglob("*"+filetype))
        self.images = [i for i in self.images if 'checkpoints' not in str(i)]
        self.transform = transform
        
        if class_to_label is None:
            class_to_label=create_label_mapping_from_paths(self.images)

        self.class_to_label = class_to_label
        self.img_size=img_size
        print(self.class_to_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_path=self.images[i]
        try:
            
            label = Path(image_path).parent.name
            image = Image.open(image_path).convert("RGB").resize(self.img_size)

            if self.transform:
                image = self.transform(image)

            encoded_label = self.class_to_label[label]
            
        except Exception as e:  # Using a more general exception class here
            print(f"An error occurred at {image_path}: {e}")
            return self.__getitem__(i+1)


        return image, encoded_label, Path(image_path).stem

# BM
class CustomImageDataset(Dataset):
    
    def __init__(self, df, transform, dic_mapping = None):
        self.df = df
        self.transform = transform
        self.class_to_label = create_label_mapping(df)
        if dic_mapping is not None: 
            self.class_to_label = {i: dic_mapping[i] for i,j in self.class_to_label.items()}
        print(self.class_to_label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, label = self.df.iloc[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(e)
            print(image_path)
            return self.__getitem__(idx+1)

        if self.transform:
            image = self.transform(image)

        encoded_label = self.class_to_label[label]

        return image, encoded_label, Path(image_path).stem
