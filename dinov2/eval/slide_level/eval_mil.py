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
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import dinov2.eval.slide_level.models.aggregators as models
from dinov2.eval.patch_level.dataset import PathImageDataset
from dinov2.eval.patch_level.general_patch_eval import save_features_and_labels
from dinov2.eval.patch_level.models.return_model import (get_models,
                                                         get_transforms)

done = time()
print("imports done in", np.round(done-start), "s")

"""
run with 
python ./dinov2/eval/slide_level/eval_mil.py --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_supcon_no_koleo/eval/training_99999/ --model_path /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_supcon_no_koleo/eval/training_99999/
"""

def get_eval_metrics(
        targets_all: Union[List[int], np.ndarray],
        preds_all: Union[List[int], np.ndarray],
        probs_all: Optional[Union[List[float], np.ndarray]] = None,
        get_report: bool = False,
        prefix: str = "",
        roc_kwargs: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
    """
    Calculate evaluation metrics and return the evaluation metrics.

    Args:
        targets_all (array-like): True target values.
        preds_all (array-like): Predicted target values.
        probs_all (array-like, optional): Predicted probabilities for each class. Defaults to None.
        get_report (bool, optional): Whether to include the classification report in the results. Defaults to True.
        prefix (str, optional): Prefix to add to the result keys. Defaults to "".
        roc_kwargs (dict, optional): Additional keyword arguments for calculating ROC AUC. Defaults to {}.

    Returns:
        dict: Dictionary containing the evaluation metrics.

    """
    bacc = balanced_accuracy_score(targets_all, preds_all)
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)

    eval_metrics = {
        f"{prefix}/acc": acc,
        f"{prefix}/bacc": bacc,
        f"{prefix}/kappa": kappa,
        f"{prefix}/weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }

    if get_report:
        eval_metrics[f"{prefix}/report"] = cls_rep

    if probs_all is not None:
        unique_classes = np.unique(targets_all)
        loss = log_loss(targets_all, probs_all, labels=unique_classes)
        eval_metrics[f"{prefix}/loss"] = loss
        roc_auc = roc_auc_score(targets_all, probs_all, labels=unique_classes, **roc_kwargs)
        
        eval_metrics[f"{prefix}/auroc"] = roc_auc
    
    for key, value in eval_metrics.items():
        print(f"{key.split('/')[-1]: <12}: {np.round(value, 4):.4f}")

    return eval_metrics


def train_evaluate_mil(
        train_data, val_data, test_data, num_classes,
        num_epochs = 20, random_state=0, dropout=0.25, n_heads=1, prefix="", wandb=False, arch="ABMIL"
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_state)

    if wandb:
        wandb.config({
            "num_epochs": num_epochs,
            "random_state": random_state,
            "dropout": dropout,
            "n_heads": n_heads,
            "lr": 0.0001,
            "weight_decay": 1.0e-05,
        })

    print(f"# train samples: {len(train_data)}, # val samples: {len(val_data)}, # test samples: {len(test_data)}")
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Initialize the model, loss function and optimizer
    in_dim = train_data[0][0].shape[-1]
    model = models.__dict__[arch](input_dim=in_dim, num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1.0e-05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train the model
    best_val_loss, best_model_weights = 1000, None
    for epoch in tqdm(range(num_epochs), desc=f"Training {arch}"):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate the model
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                val_loss = criterion(outputs, labels.to(device))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_weights = model.state_dict()      

        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')
        
    # Test the model
    model.load_state_dict(best_model_weights)
    model.eval()
    with torch.no_grad():
        test_preds_list, test_probs_list, test_labels_list = [], [], []
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, test_preds = torch.max(outputs, 1)
            test_preds_list.append(test_preds)
            test_labels_list.append(labels)

            if num_classes == 2:
                test_probs = nn.Softmax(dim=1)(outputs)[:, 1]
                roc_kwargs = {}
            else:
                test_probs = nn.Softmax(dim=1)(outputs)
                roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'} 
            test_probs_list.append(test_probs) 

    test_labels = torch.cat(test_labels_list).cpu().numpy()
    test_preds = torch.cat(test_preds_list).cpu().numpy()
    test_probs = torch.cat(test_probs_list).cpu().numpy()

    return get_eval_metrics(test_labels, test_preds, test_probs, roc_kwargs=roc_kwargs, prefix=prefix)


def save_features_and_labels(feature_extractor, transform, data_dir, save_dir, ext='.jpg'):

    print("extracting features and saving to", save_dir)
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    os.chmod(save_dir, 0o777)

    # check if features have already been extracted
    images = list(Path(data_dir).rglob(f'**/*{ext}'))
    # remove ipynb_checkpoints
    images = [img for img in images if 'ipynb_checkpoints' not in str(img)]
    features = list(Path(save_dir).rglob('*.h5'))
    if len(images) == len(features):
        print("features already extracted at", save_dir)
        return

    # create dataset from list 
    class ListDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform

            # load task configs
            with open(f'dinov2/eval/slide_level/task_configs.yaml', 'r') as f:
                task_configs = yaml.safe_load(f)
            self.dataset = args.dataset
            self.labels = pd.read_csv(task_configs[self.dataset]['csv']) if self.dataset == 'APL_AML_all' else None
            self.label_dict = task_configs[args.dataset]['label_dict']

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img = Image.open(self.data[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            
            label = self.get_labels(self.data[idx])

            return img, label, str(self.data[idx])
        
        def get_labels(self, path):
            if self.dataset == 'APL_AML_all':
                patient = path.replace(data_dir, '').split('/')[1]
                label = self.labels[self.labels['Patient_ID'] == patient]['Diagnosis']
                label = self.label_dict[label.values[0]]
            elif self.dataset == 'AML_Hehr':
                label = self.label_dict[Path(path).parent.parent.name]
            else: 
                raise NotImplementedError(f"Dataset {self.dataset} not implemented")
            
            return label

    dataset = ListDataset(images, transform)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings_list, labels_list = {}, {}
    with torch.no_grad():
        feature_extractor.eval()
        for images, labels, file_names in tqdm(dataloader):
            images = images.to(device)
            batch_features = feature_extractor(images)

            parent_names = [Path(fn).parent for fn in file_names]

            h5_filenames = [str(fn).replace(data_dir, str(save_dir)+'/') + '.h5' for fn in parent_names]
            
            for h5_filename, label, batch_feature in zip(h5_filenames, labels, batch_features):
                if h5_filename not in embeddings_list.keys():
                    embeddings_list[h5_filename] = []
                    labels_list[h5_filename] = label
                embeddings_list[h5_filename].append(batch_feature.cpu().numpy())
                assert labels_list[h5_filename] == label, "labels are not the same for the same patient"
    
    for h5_filename, embeddings in embeddings_list.items():
        embeddings = np.stack(embeddings)
        label = labels_list[h5_filename]
        save_h5(h5_filename, embeddings, label)


def save_h5(h5_filename, embeddings, label):
    Path(h5_filename).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_filename, "w") as hf:
        hf.create_dataset("features", data=embeddings)
        hf.create_dataset("labels", data=label)


def eval_task(dataset, feature_dir, folds):
    # load task configs
    with open(f'dinov2/eval/slide_level/task_configs.yaml', 'r') as f:
        task_configs = yaml.safe_load(f)
    
    class H5Dataset(Dataset):
        def __init__(self, root):
            self.data = list(Path(root).rglob('**/*.h5'))

            # print('Loading labels ...')
            # self.labels = [h5_file.parent.parent.name for h5_file in self.data]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            with h5py.File(self.data[idx], 'r') as hf:
                features = hf['features'][()]
                label = hf['labels'][()]

            return features, label

    if dataset == 'AML_Hehr':
        # 5-fold cross val on training data
        train_val_dataset = H5Dataset(Path(feature_dir) / 'train')
        train_val_labels = [path.parent.name for path in train_val_dataset.data]
        test_dataset = H5Dataset(Path(feature_dir) / 'test')

        splits = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        results = []
        for fold, (train_idx, val_idx) in enumerate(splits.split(train_val_dataset, train_val_labels)):
            train_dataset = Subset(train_val_dataset, train_idx)
            val_dataset = Subset(train_val_dataset, val_idx)

            res = train_evaluate_mil(train_dataset, val_dataset, test_dataset, num_classes)
            results.append(res)
        
        print("====================================")
        print(f'Final results averaged over {folds} folds for {dataset}')
        results_mean = {key: np.mean([result[key] for result in results]) for key in results[0].keys()}
        results_std = {key: np.std([result[key] for result in results]) for key in results[0].keys()}
        for keys, values in results_mean.items():
            print(f"{keys.split('/')[-1]: <12}: {np.round(values, 4):.4f} ± {np.round(results_std[keys], 4):.4f}")

    else:  
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a WBC MIL model.')
    parser.add_argument('--dataset', 
                        type=str, 
                        help='name of the dataset for evaluation',
                        default='AML_Hehr')
    # parser.add_argument('--checkpoint', 
    #                     type=str, 
    #                     help='checkpoint to evaluate')   
    parser.add_argument(
        "--img_size",
        help="size of image to be used",
        default=224,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
    )
                        

    args = parser.parse_args()

    args.checkpoint = '/lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-S.pth'
    checkpoint_root = Path('/lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-S') # Path(args.checkpoint).parent.parent
    model_name = 'dinov2_vits14'
    dataset = 'AML_Hehr'
    folds = 5
    wandb.init(project="histo-collab", entity="DinoBloomv2-MIL-eval", name=checkpoint_root.name+'_'+dataset, mode='disabled') 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_epochs = 20

    num_classes = 5
    batch_size = 1

    # extract embeddings
    feature_dir = checkpoint_root / 'features' / dataset
    feature_dir.mkdir(exist_ok=True, parents=True, mode=0o777)

    # Load the model
    if model_name in ["owkin","resnet50","resnet50_full","remedis","imagebind"]:
        sorted_paths=[None]
    elif model_name in ["retccl","ctranspath"]:
        sorted_paths=[Path(args.checkpoint)]
    elif Path(args.checkpoint).is_file():
        sorted_paths = [Path(args.checkpoint)]
    else:
        sorted_paths = list(Path(args.checkpoint).rglob("*teacher_checkpoint.pth"))

    # load task configs
    with open(f'dinov2/eval/slide_level/task_configs.yaml', 'r') as f:
        task_configs = yaml.safe_load(f)

    for checkpoint in sorted_paths:
        feature_extractor = get_models(model_name, args.img_size, saved_model_path=checkpoint)
        transform = get_transforms(model_name, args.img_size)

        save_features_and_labels(feature_extractor, transform, task_configs[args.dataset]['root'], feature_dir, ext=task_configs[dataset]['ext'])

        if len(sorted_paths)>1:
            sorted_paths = sorted(sorted_paths, key=sort_key)

        results = eval_task(dataset, feature_dir, folds=folds)

        results_mean = {key: np.mean([result[key] for result in results]) for key in results[0].keys()}
        results_std = {key: np.std([result[key] for result in results]) for key in results[0].keys()}

        wandb.log(results_mean)
        print("====================================")
        print(f'Final results averaged over {folds} folds for {dataset}')
        for keys, values in results_mean.items():
            print(f"{keys.split('/')[-1]: <12}: {np.round(values, 4):.4f} ± {np.round(results_std[keys], 4):.4f}")
        
    #     task = list(results_mean.keys())[0].split("/")[0]
    #     results[task] = {}
    #     for key, value in results_mean.items():
    #         metric = key.split("/")[-1]
    #         results[task][f"{metric}_mean"] = value
    #         results[task][f"{metric}_std"] = results_std[key]
    
    # pd.DataFrame(results).to_csv(f"{cfg.base_dir}/results_{cfg.arch}.csv")
    # print(f"results saved to {cfg.base_dir}/results_{cfg.arch}.csv")


    
