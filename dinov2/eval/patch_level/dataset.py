"""
taken from feature_extraction branch from HistoBistro
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from dinov2.eval.patch_level.utils import create_label_mapping, create_label_mapping_from_paths


class WBCMILDataset(Dataset):
    def __init__(self, data_path, transform):
        self.transform = transform
        self.images = []
        
        clses = os.listdir(data_path)
        for cls in clses:
            patients = os.listdir(os.path.join(data_path, cls))
            for patient in patients:
                cells = os.listdir(os.path.join(data_path, cls, patient))
                for cell in cells:
                    if cell.lower().endswith('.tif'):
                        cell_path = os.path.join(data_path, cls, patient, cell)
                        self.images.append(cell_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path


class WbcAttDataset(Dataset):

    def __init__(self, image_path, label_file, transform,img_size=(224,224)):
        self.image_base_path = image_path
        self.transform = transform
        self.label_file = pd.read_csv(label_file)
        self.img_size=img_size
        #self.folder_routing={"MO":"monocyte","BA":"basophil","ERB":"erythroblast","MMY":"metamyelocyte", "LY":"lymphocyte_typical",
        #"MY":"myelocyte","BNE":"neutrophil_band","SNE":"neutrophil_segmented", "PMY":"promyelocyte"}
        self.folder_routing={"MO":"monocyte","BA":"basophil","ERB":"erythroblast","MMY":"MMY", "LY":"lymphocyte",
        "MY":"MY","BNE":"BNE","SNE":"SNE", "PMY":"PMY","NEUTROPHIL":"NEUTROPHIL","IG":"IG","EO":"eosinophil"}
        self.wbc_att_class_mapping = {
            'label': {
                'Neutrophil': 0,
                'Eosinophil': 1,
                'Basophil': 2,
                'Lymphocyte': 3,
                'Monocyte': 4
            },
            'cell_size': {
                'big': 0,
                'small': 1
            },
            'cell_shape': {
                'round': 0,
                'irregular': 1
            },
            'nucleus_shape': {
                'unsegmented-band': 0,
                'unsegmented-round': 1,
                'segmented-multilobed': 2,
                'segmented-bilobed': 3,
                'irregular': 4,
                'unsegmented-indented': 5
            },
            'nuclear_cytoplasmic_ratio': {
                'low': 0,
                'high': 1
            },
            'chromatin_density': {
                'densely': 0,
                'loosely': 1
            },
            'cytoplasm_vacuole': {
                'no': 0,
                'yes': 1
            },
            'cytoplasm_texture': {
                'clear': 0,
                'frosted': 1
            },
            'cytoplasm_colour': {
                'light blue': 0,
                'blue': 1,
                'purple blue': 2
            },
            'granule_type': {
                'small': 0,
                'round': 1,
                'coarse': 2,
                'nil': 3
            },
            'granule_colour': {
                'pink': 0,
                'purple': 1,
                'red': 2,
                'nil': 3
            },
            'granularity': {
                'yes': 0,
                'no': 1
            }
        }
    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, i):
        entry=self.label_file.iloc[i]
        labels = entry.drop(['img_name', 'path'])
        filename=entry["img_name"]
        image_path=Path(self.image_base_path)/self.folder_routing[filename.split("_")[0]]/filename
        try:

            image = Image.open(image_path).convert("RGB").resize(self.img_size)
            if self.transform:
                image = self.transform(image)

            encoded_labels = [self.wbc_att_class_mapping[key][value] for key, value in labels.items()]
            
        except Exception as e:  # Using a more general exception class here
            print(f"An error occurred at {image_path}: {e}")
            return self.__getitem__(i+1)

        return image, encoded_labels, Path(image_path).stem
    


class PathImageDataset(Dataset):
    def __init__(self, image_path, transform,class_to_label=None,filetype=".tiff",img_size=(224,224)):
        self.images = list(Path(image_path).rglob("*"+filetype))
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

class CustomImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        self.class_to_label = create_label_mapping(df)
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

