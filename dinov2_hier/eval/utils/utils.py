import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os



def create_label_mapping(df):
    """
    Creates a dictionary mapping each unique class label in the dataframe to an integer.

    Parameters:
    - df: pandas DataFrame containing a column 'Label' with class labels.

    Returns:
    - A dictionary mapping each unique label to an integer, starting from 0.
    """
    # Get unique labels and sort them
    unique_labels = sorted(df["Label"].unique())

    # Create mapping
    label_to_int = {label: index for index, label in enumerate(unique_labels)}

    return label_to_int

def create_label_mapping_from_paths(image_paths):
    """
    Creates a dictionary mapping each unique class label, derived from the parent folder name, to an integer.

    Parameters:
    - image_paths: List of strings, where each string is the file path of an image.

    Returns:
    - A dictionary mapping each unique label to an integer, starting from 0.
    """
    # Extract class labels from parent folder names
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]

    # Get unique labels and sort them
    unique_labels = sorted(set(labels))

    # Create mapping
    label_to_int = {label: index for index, label in enumerate(unique_labels)}

    return label_to_int

