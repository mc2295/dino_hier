
import pandas as pd

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


class CytologiaDatasetUnSup(VisionDataset):
    def __init__(
        self,
        *,
        df_path: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__(df_path, transforms, transform, target_transform)
        df = pd.read_csv(df_path)
        self.df = df.loc[df['is_valid'] == False]
        self.true_len=len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        

        row = self.df.iloc[index]
        img_path = row['NAME']
        try:
            image = Image.open( img_path).convert(mode="RGB")
        except Exception as e: 
            print('cannot load image ', img_path)
            return self.__getitem__(index + 1)
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.Tensor(0)


    def __len__(self) -> int:
        # assert len(entries) == self.split.length
        return len(self.df)


class CytologiaDatasetSup(VisionDataset):
    def __init__(
        self,
        *,
        df_path: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
        is_valid: bool = False
    ) -> None:
        super().__init__(df_path, transforms, transform, target_transform)
        df = pd.read_csv(df_path)
        self.df = df.loc[df['is_valid'] == False]
        self.true_len=len(self.df)
        unique_labels = sorted(self.df["class"].unique())
        self.target_transform = target_transform
        # Create mapping
        self.label_to_int = {label: index for index, label in enumerate(unique_labels)}
        print("number of sup samples", len(self.df))

    def __getitem__(self, index: int):
        
        row = self.df.iloc[index]
        label_cat =  row['class']
        #label = self.label_to_int[label_cat]
        label = self.target_transform(label_cat)
        img_path = row['NAME']
        #x1, x2, y1, y2 = row['x1'], row['x2'], row['y1'], row['y2']
        try:
            image = self._load_img_(img_path)
        except Exception as e: 
            print('cannot load image ', img_path)
            return self.__getitem__(index + 1)
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label, img_path, label_cat

    def _load_img_(self, img_path):
        img = Image.open( img_path).convert(mode="RGB")

        #if p>0:
        #img = img.crop((x1,y1,x2,y2))
        return img

    def __len__(self) -> int:
        # assert len(entries) == self.split.length
        return len(self.df)