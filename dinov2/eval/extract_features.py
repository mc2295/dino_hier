import os
import h5py
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import argparse
from pathlib import Path
import yaml

# Assume get_models is a provided function to load the feature extractor
from dinov2.eval.patch_level.models.return_model import (get_models,
                                                         get_transforms)
class ImageDataset(Dataset):
    def __init__(self, image_paths, label_dict, transform=None):
        self.image_paths = image_paths
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try: 
            image = Image.open(img_path).convert(mode="RGB")
        except Exception as e:
            print(f"can not read image for sample {idx, e, img_path}")
            # initialize empty image
            image = torch.zeros((3, 224, 224), dtype=torch.float32)
            return image, img_path, -torch.inf
        if self.transform:
            image = self.transform(image)
        label = Path(img_path).parent.name
        label = self.label_dict[label] if label in self.label_dict else -1
        return image, img_path, label

def extract_features(txt_file, output_file, model_name, img_size, checkpoint, label_dict, batch_size=32):
    # Load the feature extractor model
    feature_extractor = get_models(model_name, img_size, saved_model_path=checkpoint)
    feature_extractor = feature_extractor.cuda()
    feature_extractor.eval()

    transform = get_transforms(model_name, args.img_size)

    # Read the image paths from the txt file
    with open(txt_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    # Create a dataset and data loader
    dataset = ImageDataset(image_paths, label_dict, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    features = []
    img_paths = []
    labels = []
    chunk_counter = 0
    for images, paths, label in tqdm(dataloader):
        with torch.no_grad():
            # Pass the batch of images through the model to extract features
            batch_features = feature_extractor(images.cuda())

        # filter out invalid images
        batch_features = batch_features[~(label == -torch.inf)]
        paths = [p for p, l in zip(paths, label) if l != -torch.inf]
        label = [l for l in label if l != -torch.inf]

        # Convert features to numpy and store them
        features.append(batch_features.cpu().numpy())
        img_paths.extend(paths)
        labels.extend(label)

        # Save in chunks if necessary
        if len(img_paths) >= 32768:
            # add counter to output file name, such that current counter is used and old counters are discared
            if chunk_counter > 0:
                output_file = output_file.replace(f'_{chunk_counter-1}.h5', f'_{chunk_counter}.h5')
            else:
                output_file = output_file.replace('.h5', f'_{chunk_counter}.h5')
            save_as_h5(features, img_paths, labels, output_file)
            features = []
            img_paths = []
            labels = []
            chunk_counter += 1

    # Save any remaining features
    if len(img_paths) > 0:
        save_as_h5(features, img_paths, labels, output_file)
    print(f"Features saved to {output_file}")

def save_as_h5(features, img_paths, labels, output_file):
    with h5py.File(output_file, 'a') as f:
        f.create_dataset('features', data=np.vstack(features))
        f.create_dataset('img_paths', data=img_paths)
        f.create_dataset('labels', data=labels)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract features from images listed in a text file.")
    parser.add_argument('txt_file', type=str, help="Path to the text file containing image paths.")
    parser.add_argument('output_dir', type=str, help="Directory where the output file will be saved.")
    parser.add_argument('model_name', type=str, help="Name of the model to use for feature extraction.")
    parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint file.")
    parser.add_argument('--img_size', type=int, default=224, help="Image size expected by the model.")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size for processing images.")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    config = '/lustre/groups/shared/users/peng_marr/DinoBloomv2/configs/custom_2_alternating.yaml'
    with open(config, 'r') as f:
        label_dict = yaml.load(f, Loader=yaml.FullLoader)['label_dict']

    dataset_name = Path(args.txt_file).stem
    output_file = os.path.join(args.output_dir, f'{dataset_name}_embeddings.h5')
    extract_features(args.txt_file, output_file, args.model_name, args.img_size, args.checkpoint, label_dict, args.batch_size)

