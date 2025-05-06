from sklearn.model_selection import StratifiedKFold
import numpy as np
import math

def create_stratified_folds(labels):
    """
    Splits indices into 5 stratified folds based on the provided labels,
    returning indices for train and test sets for each fold.
    
    Args:
    - labels (array-like): Array or list of labels to be used for creating stratified folds.
    
    Returns:
    - A list of tuples, each containing two arrays: (train_indices, test_indices) for each fold.
    """
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5)
    
    # Prepare for stratified splitting
    folds = []
    for train_index, test_index in skf.split(X=np.arange(len(labels)), y=labels):
        folds.append((train_index, test_index))
    
    return folds

def merge_list_dicts(list_dict, new_dict, path=None):
    if path is None:
        path = []
    for key, value in new_dict.items():
        new_path = path + [key]
        if isinstance(value, dict):
            list_dict[key] = merge_list_dicts(list_dict.get(key, {}), value, new_path)
        elif isinstance(value, str):
            list_dict[key] = value  # Directly assign the string
        else:
            if key not in list_dict or isinstance(list_dict[key], str):
                list_dict[key] = []
            list_dict[key].append(value)
    return list_dict

def compute_mean_std(current_dict):
    mean_dict = {}
    std_dict = {}
    for key, value in current_dict.items():
        if isinstance(value, dict):
            mean_subdict, std_subdict = compute_mean_std(value)
            mean_dict[key] = mean_subdict
            std_dict[key] = std_subdict
        elif isinstance(value, str):
            mean_dict[key] = value
            std_dict[key] = value  # You could also set this to None or skip it if preferred
        else:
            n = len(value)
            mean = sum(value) / n
            variance = sum((x - mean) ** 2 for x in value) / n
            std = math.sqrt(variance)
            mean_dict[key] = mean
            std_dict[key] = std
    return mean_dict, std_dict

def compute_mean_bootstrap(current_dict):
    mean_dict = {}
    bs_dict = {}
    alpha = 0.05
    for key, value in current_dict.items():
        if isinstance(value, str):
            mean_dict[key] = value
            bs_dict[key] = value 
        else:
            n = len(value)
            mean = sum(value) / n
            bs_int = np.percentile(value, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            bs_value = (bs_int[1] - bs_int[0])/2
            mean_dict[key] = mean
            bs_dict[key] = bs_value
    return mean_dict, bs_dict


def mean_std_dicts(dict_list):
    list_dict = {}
    for d in dict_list:
        merge_list_dicts(list_dict, d)
    return compute_mean_std(list_dict)

def mean_bs_dicts(dict_list):
    list_dict = {}
    for d in dict_list:
        merge_list_dicts(list_dict, d)
    return compute_mean_bootstrap(list_dict)