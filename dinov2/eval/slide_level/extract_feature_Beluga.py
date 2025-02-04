from time import time

start = time()
import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from tqdm import tqdm 

from PIL import Image
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from dinov2.eval.patch_level.models.return_model import (get_models,
                                                         get_transforms)


def save_features_and_labels_Beluga(task_configs, save_dir, checkpoint, args):

    print("extracting features and saving to", save_dir)
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    os.chmod(save_dir, 0o777)

    feature_extractor = get_models(args.model_name, args.img_size, saved_model_path=checkpoint)
    transform = get_transforms(args.model_name, args.img_size, saved_model_path=checkpoint)

    # create dataset from list 
    class ListDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.data[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
            except Exception as e:
                print(f"Error: {e}, idx: {idx}, path: {self.data[idx]}")
                img = torch.zeros(3, 224, 224)

            return img
        

    # get all images for features extraction
    patient_csv = pd.read_csv(task_configs[args.dataset]['csv'])
    label_dict = task_configs[args.dataset]['label_dict']
    ext = task_configs[args.dataset]['ext']
    data_dir = task_configs[args.dataset]['root']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(list(save_dir.glob("*.h5"))) == len(patient_csv):
        print("embeddings have already been extracted...")
        return

    for patient,label in tqdm(zip(patient_csv['Patient_ID'],patient_csv['Diagnosis'])):
        patient_path = Path(data_dir) / patient
        images = list(Path(patient_path).rglob(f'*{ext}'))
        images = [img for img in images if 'ipynb_checkpoints' not in str(img)]

        label = label_dict[label]

        dataset = ListDataset(images, transform)
        dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

        h5_filename = str(patient_path).replace(data_dir, str(save_dir)+'/') + '.h5'

        if os.path.isfile(h5_filename):  # check if file exists already
            continue
        else:
            embeddings_list = []
            with torch.no_grad():
                feature_extractor.eval()
                for images in dataloader:
                    images = images.to(device)
                    if args.model_name.lower()=="conch":
                        batch_features=feature_extractor.encode_image(images, proj_contrast=False, normalize=False)
                    else:
                        batch_features = feature_extractor(images)

                    embeddings_list.append(batch_features.cpu().numpy())
                
            if not embeddings_list:
                print(f"No embeddings generated for patient: {patient}")
                continue
            embeddings = np.concatenate(embeddings_list,axis=0)
            save_h5(h5_filename, embeddings, label)


def save_h5(h5_filename, embeddings, label):
    Path(h5_filename).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_filename, "w") as hf:
        hf.create_dataset("features", data=embeddings)
        hf.create_dataset("labels", data=label)