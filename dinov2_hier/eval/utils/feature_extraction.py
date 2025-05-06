
from tqdm import tqdm
import os
import torch
import h5py
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path

def process_file(file_name):
    with h5py.File(file_name, "r") as hf:
        features = torch.tensor(hf["features"][:]).tolist()
        label = int(hf["labels"][()])
    return features, label, Path(file_name).name

def save_features_and_labels(feature_extractor, dataloader, save_dir,dataset_len,model_name):

    print("extracting features..")
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    os.chmod(save_dir, 0o777)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        feature_extractor.eval()

        for images, labels, names in tqdm(dataloader):
            images = images.to(device)
            if model_name.lower()=="conch":
                batch_features=feature_extractor.encode_image(images, proj_contrast=False, normalize=False)
            else:
                batch_features = feature_extractor(images)

            labels_np = labels.numpy()

            for img_name, img_features, img_label in zip(names, batch_features, labels_np):
                h5_filename = os.path.join(save_dir, f"{img_name}.h5")

                with h5py.File(h5_filename, "w") as hf:
                    hf.create_dataset("features", data=img_features.cpu().numpy())
                    hf.create_dataset("labels", data=img_label)



def get_data(all_data):
    # Define the directories for train, validation, and test data and labels

    # Load training data into dictionaries
    features, labels, filenames = [], [], []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_name) for file_name in all_data]

        for i, future in tqdm(enumerate(futures), desc="Loading data"):
            feature, label,filename = future.result()
            features.append(feature)
            labels.append(label)
            filenames.append(filename)

    # Convert the lists to NumPy arrays
    features = np.array(features)
    labels = np.array(labels).flatten()
    # Flatten test_data
    features = features.reshape(features.shape[0], -1)  # Reshape to (n_samples, 384)

    return features, labels, filenames

