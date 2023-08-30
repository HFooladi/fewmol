import numpy as np
from typing import Dict, List
from fewmol.data import FSMOLDataset


def get_split(chembl_id: str, requested_training_size: int = 8, random_seed=42) -> Dict:
    """
    Split the dataset into training and testing set
    :param chembl_id: ChEMBL ID of the molecule
    :param requested_training_size: Number of molecules in the training set
    :param random_seed: Random seed for reproducibility

    :return: A dictionary of the split for the given chembl_id
    {'chembl_id': {'requested_training_size': {'train': [train_index], 'test': [test_index]}}}
    """
    data = FSMOLDataset(chembl_id)
    dataset_index = np.arange(len(data))
    assert requested_training_size < len(
        dataset_index
    ), "Requested training size is larger than the dataset size"
    np.random.seed(random_seed)
    train_index = np.random.choice(dataset_index, size=requested_training_size, replace=False)
    test_index = dataset_index[~np.isin(dataset_index, train_index)]
    print(f"Train size: {len(train_index)}")
    print(f"Test size: {len(test_index)}")
    split_idx = {"train": train_index.tolist(), "test": test_index.tolist()}
    return {f"{chembl_id}": {f"{requested_training_size}": split_idx}}


def get_split_multiple_size(
    chembl_id: str, training_size_list: List = [8, 16, 32], random_seed=42
) -> Dict:
    """
    Split the dataset into training and testing set for each testing size
    :param chembl_id: ChEMBL ID of the molecule
    :param training_size_list: List of training size
    :param random_seed: Random seed for reproducibility

    :return: A dictionary of the split for the given chembl_id
    """
    my_split = {}
    my_split[chembl_id] = {}
    for size in training_size_list:
        my_split[chembl_id].update(get_split(chembl_id, size)[chembl_id])

    assert len(my_split[chembl_id]) == len(
        training_size_list
    ), "The number of testing size list is not equal to the number of splits"
    return my_split


def get_split_multiple_assay(
    chembl_ids: List, training_size_list=[8, 16, 32], random_seed=42
) -> Dict:
    """
    Split the dataset into training and testing set for each testing size
    :param chembl_ids: List of ChEMBL IDs of the molecules
    :param training_size_list: List of training size

    :return: A dictionary of the split for the given chembl_id
    """
    my_split = {}
    for chembl_id in chembl_ids:
        my_split.update(get_split_multiple_size(chembl_id, training_size_list, random_seed))

    assert len(my_split) == len(
        chembl_ids
    ), "The number of ChEMBL IDs is not equal to the number of splits"
    return my_split
