from typing import Dict, List, Tuple

import pandas as pd
import torch

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def select_training_assays(training_lookup: Dict, distance_dict: Dict) -> Tuple[torch.Tensor, List[str]]:
    distance = []
    train_assays = []
    for i, assay in enumerate(distance_dict['train_chembl_ids']):
        if assay in training_lookup['train']:
            distance.append(distance_dict['distance_matrices'][i])
            train_assays.append(assay)

    d_mat = torch.stack(distance, dim=0)
    return d_mat, train_assays


def select_test_assay(chembl_id: str, distance_mat: torch.Tensor, distance_dict: Dict) -> torch.Tensor:
    for i, assay in enumerate(distance_dict['test_chembl_ids']):
        if assay == chembl_id:
            test_id = i
            break
    
    x = distance_mat[:, test_id]
    return x


def create_distance_dataframe(distance_mat: torch.Tensor, assays: List) -> pd.DataFrame:
    # Filling NaN with the mean of the matrix
    distance_mat = torch.nan_to_num(distance_mat, nan=distance_mat.nanmean())
    # Creating Dataframe with the distance matrix
    df = pd.DataFrame({"chemical_distance": distance_mat,  "assay_id": assays})
    df.set_index('assay_id', inplace=True)
    return df