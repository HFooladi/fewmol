from typing import Dict, List, Tuple

import pandas as pd
import torch


def normalize(x):
    """
    Normalize a tensor between 0 and 1. output have values between 0 and 1.
    """
    return (x - x.min()) / (x.max() - x.min())


def select_training_assays(
    lookup_table: Dict, distance_dict: Dict
) -> Tuple[torch.Tensor, List[str]]:
    """
    Select the training assays (that are available in lookup table)
    from the distance dictionary
    
    Args:
        lookup_table (Dict): lookup table with train and test assay ids
        distance_dict (Dict): distance dictionary with train and test assay ids
    Returns:
        d_mat (torch.Tensor): distance matrix of training assays
        train_assays (List[str]): list of training assay ids
    """
    assert (
        "train_chembl_ids" in distance_dict.keys()
    ), f"{distance_dict} must have train_chembl_ids key"
    assert (
        "distance_matrices" in distance_dict.keys()
    ), f"{distance_dict} must have distance_matrices key"
    assert "train" in lookup_table.keys(), f"{lookup_table} must have train key"
    distance = []
    train_assays = []
    for i, assay in enumerate(distance_dict["train_chembl_ids"]):
        if assay in lookup_table["train"]:
            distance.append(distance_dict["distance_matrices"][i])
            train_assays.append(assay)

    d_mat = torch.stack(distance, dim=0)
    return d_mat, train_assays


def select_test_assay(
    chembl_id: str, distance_mat: torch.Tensor, distance_dict: Dict
) -> torch.Tensor:
    assert (
        "test_chembl_ids" in distance_dict.keys()
    ), f"{distance_dict} must have test_chembl_ids key"
    assert distance_mat.ndim == 2, f"distance_mat must be 2D tensor, got {distance_mat.ndim}"

    try:
        test_id = distance_dict["test_chembl_ids"].index(chembl_id)
    except ValueError:
        raise ValueError(f"Assay {chembl_id} is not in the test set.")

    x = distance_mat[:, test_id]
    return x


def create_distance_dataframe(distance_mat: torch.Tensor, assays: List) -> pd.DataFrame:
    assert distance_mat.ndim == 1
    assert distance_mat.shape[0] == len(assays)
    # Filling NaN with the mean of the matrix
    distance_mat = torch.nan_to_num(distance_mat, nan=distance_mat.nanmean())
    # Creating Dataframe with the distance matrix
    df = pd.DataFrame({"distance": distance_mat, "assay_id": assays})
    df.set_index("assay_id", inplace=True)
    return df


def merge_distance(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two distance dataframes based on index
    """
    dist_df = pd.merge(df1, df2, left_index=True, right_index=True)
    dist_df.columns = ["distance_x", "distance_y"]

    return dist_df


def total_distance(df, operation="sum"):
    """
    Calculate the total distance between two distance matrices
    """
    if operation == "sum":
        return df["distance_x"] + df["distance_y"]
    elif operation == "mean":
        return (df["distance_x"] + df["distance_y"]) / 2
    elif operation == "max":
        return df[["distance_x", "distance_y"]].max(axis=1)
    elif operation == "chemical":
        return df["distance_x"]
    elif operation == "protein":
        return df["distance_y"]

    else:
        raise ValueError(f"Operation {operation} is not supported. Use supported operations")


def final_distance_df(
    lookup_table: Dict,
    chemical_distance_dict: Dict,
    protein_distance_dict: Dict,
    chembl_id: str,
    operation: str = "sum",
) -> pd.DataFrame:
    d_mat, train_assays = select_training_assays(lookup_table, chemical_distance_dict)
    d_mat_protein, train_assays_protein = select_training_assays(
        lookup_table, protein_distance_dict
    )

    d_mat = select_test_assay(chembl_id, d_mat, chemical_distance_dict)
    d_mat_protein = select_test_assay(chembl_id, d_mat_protein, protein_distance_dict)

    chemical_df = create_distance_dataframe(d_mat, train_assays)
    protein_df = create_distance_dataframe(d_mat_protein, train_assays_protein)

    chemical_df["distance"] = normalize(chemical_df["distance"])
    protein_df["distance"] = normalize(protein_df["distance"])

    dist_df = merge_distance(chemical_df, protein_df)
    df = total_distance(dist_df, operation=operation)
    df = pd.DataFrame(df)
    df.columns = ["distance"]

    return df
