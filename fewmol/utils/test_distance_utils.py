import torch
from fewmol.utils.distance_utils import create_distance_dataframe


def test_create_distance_dataframe():
    distance_mat = torch.tensor([1.0, 0.5, 3.5])
    distance_mat_with_nan = torch.tensor([1.0, 0.5, float("nan")])
    assays = ["assay_a", "assay_b", "assay_c"]
    df = create_distance_dataframe(distance_mat, assays)
    df_with_nan = create_distance_dataframe(distance_mat_with_nan, assays)

    assert df.shape == (3, 1)
    assert df.index.tolist() == ["assay_a", "assay_b", "assay_c"]

    assert df_with_nan.shape == (3, 1)
