# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

logger = logging.getLogger("dinov2")



def arrange_files(file_paths):
    # Group files by their parent folder
    grouped_files = defaultdict(list)
    for file_path in file_paths:
        parent_folder = Path(file_path).parent.name
        grouped_files[parent_folder].append(file_path)

    # Create a balanced ordering of files
    balanced_ordering = []
    # Use itertools.zip_longest for round-robin style iteration
    for group in itertools.zip_longest(*grouped_files.values()):
        # Filter out 'None' in case some groups are smaller than others
        balanced_ordering.extend(filter(None, group))

    return balanced_ordering


class HemaStandardDataset(VisionDataset):
    def __init__(
        self,
        *,
        root: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        domain_target_transform: Optional[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.patches = []
        self.domain_target_transform = domain_target_transform

        all_dataset_files = Path(root).glob("*.txt")
        name_datasets = ['bm_train_patches.txt', 'mll_mil_train_patches.txt', 'matek_patches.txt', 'raabin_train_patches.txt', 
                            'warthog_patches.txt', 'lisc_refactor_patches.txt', 'ssl_seg_patches_all.txt', 'chula_patches.txt',
                            'bccd_patches.txt', 'blood_cell_detection_kaggle_patches.txt', 'nuclick_hema_patches.txt',
                            'marr_mll_patches.txt']
        dataset_list = {root + '/' + i for i in name_datasets}

        for dataset_file in all_dataset_files:
            print("Loading ", dataset_file)

            if str(dataset_file) in dataset_list:
                with open(dataset_file, 'r') as file:
                    content = file.read()
                file_list = content.splitlines()
                self.patches.extend(file_list)
        self.true_len = len(self.patches)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image , filepath = self.get_image_data(index)
        except Exception as e:
            adjusted_index=index%self.true_len
            filepath = self.patches[adjusted_index]
            print(f"can not read image for sample {index, e,filepath}")
            return self.__getitem__(index + 1)

        target = self.get_target(index)

        domain_label = self.get_domain_label(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, filepath, domain_label

    def get_image_data(self, index: int, dimension=224) -> Image:
        # Load image from jpeg file
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        patch = Image.open(filepath).convert(mode="RGB").resize((dimension,dimension),Image.Resampling.LANCZOS)
        return patch, filepath

    def get_target(self, index: int) -> torch.Tensor:
        # Get the label from the file path                
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        label = Path(filepath).parent.name

        return label
    
    def get_domain_label(self, index: int) -> torch.Tensor:
        # Get the label from the file path
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        domain = Path(filepath).parts[5]
        if domain == "qscd01":
            domain = Path(filepath).parts[9] if Path(filepath).parts[8] == "_Domains" else Path(filepath).parts[7]
        elif domain == "patches":
            domain = Path(filepath).parts[6]
        return self.domain_target_transform(domain)

    def __len__(self) -> int:
        # assert len(entries) == self.split.length
        return 120000000
    

class HemaAlternatingDataset(VisionDataset):
    def __init__(
        self,
        *,
        root: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        domain_target_transform=None,
        shuffle: bool = False,
        label_file: str = None,
        label_confidence_cut: float = 0.7
        
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.domain_target_transform = domain_target_transform
        patches_unlabeled = self.create_patch_list(Path(root)/"unlabeled")
        patches_labeled_i = self.create_patch_list(Path(root)/"labeled_i")
        patches_labeled_ii=[]

        if label_file is not None:
            df=pd.read_csv(label_file)
            df = df[df['primary_class'] != '2-MISC']
            self.label_file = df[df['primary_probability'] >= label_confidence_cut]
            self.label_file.set_index('object_key', inplace=True)
            more_labeled_patches,more_unlabeled_patches = self.split_dataset(Path(root)/"labeled_ii/beluga_train_patches.txt")
            all_patches=[patches_unlabeled+more_unlabeled_patches,patches_labeled_i+more_labeled_patches]
        else:
            self.label_file = None
            all_patches=[patches_unlabeled,patches_labeled_i]
        

        self.patches=[a for a in all_patches if len(a)>1]
        self.dataset_sizes=[len(a) for a in self.patches]
        self.num_datasets=len(self.patches)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        dataset_index = index % self.num_datasets #can be changed to make different schedule
        index_in_dataset = int(index / self.num_datasets) % self.dataset_sizes[dataset_index]
        filepath = self.patches[dataset_index][index_in_dataset]

        try:
            image = self.get_image_data(filepath)

        except Exception as e:
            print(f"can not read image for sample {e,filepath}")
            return self.__getitem__(index + 1)

        target = self.get_target(filepath)
        domain_label = self.get_domain_label(filepath)

        if self.transforms is not None:
            image, target = self.transforms(image, target)


        return image, target, filepath, domain_label
    
    def get_domain_label(self, filepath:str) -> torch.Tensor:
        # Get the label from the file path
        domain = Path(filepath).parts[5]
        if domain == "qscd01":
            domain = Path(filepath).parts[9] if Path(filepath).parts[8] == "_Domains" else Path(filepath).parts[7]

        return self.domain_target_transform(domain)
    
    def create_patch_list(self, root:Path):

        files=root.glob("*.txt")
        patches=[]

        for dataset_file in files:
            print("Loading ", dataset_file)
            with open(dataset_file, 'r') as file:
                content = file.read()
            file_list = content.splitlines()
            patches.extend(file_list)
        return patches

    def get_image_data(self, filepath, dimension=224) -> Image:
        # Load image from jpeg file
        patch = Image.open(filepath).convert(mode="RGB").resize((dimension,dimension),Image.Resampling.LANCZOS)
        return patch

    def get_target(self, filepath) -> torch.Tensor:
        # Get the label from the file path
        if "BELUGA" in filepath and self.label_file is not None:
            split_path = str(filepath).split("/")
            cell_id = f"{split_path[-2]}/{split_path[-1]}"
            
            # Error handling: return "no_label" if cell_id is not found
            label = self.label_file['primary_class'].get(cell_id, "no_label")
        else:
            label = Path(filepath).parent.name
        return label

    def split_dataset(self, file_paths_file):

        # Read the file paths from the second file
        with open(file_paths_file, 'r') as file:
            file_paths = [line.strip() for line in file]

        # Initialize lists to store file paths
        labeled_files = []
        unlabeled_files = []

        # Loop through file paths and check if the respective cell_id is in df_filtered
        for file_path in file_paths:
            split_path = file_path.split('/')
            cell_id = f"{split_path[-2]}/{split_path[-1]}"  # Extracting cell_id

            if cell_id in self.label_file.index:
                labeled_files.append(file_path)
            else:
                unlabeled_files.append(file_path)

        # Return both lists
        return labeled_files, unlabeled_files


    def __len__(self) -> int:
        # assert len(entries) == self.split.length
        return 120000000
    