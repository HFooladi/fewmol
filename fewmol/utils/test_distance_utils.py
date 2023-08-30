import pytest
import torch
from fewmol.utils import select_training_assays, select_test_assay, create_distance_dataframe


@pytest.fixture
def distance_dict():
    train_assays = ["assay_a", "assay_b", "assay_c", "assay_d"]
    test_assay = ["assay_e", "assay_f"]
    distance_matrix = torch.arange(8).reshape(4, 2)
    return {
        "train_chembl_ids": train_assays,
        "test_chembl_ids": test_assay,
        "distance_matrices": distance_matrix,
    }


@pytest.fixture
def lookup_table():
    train_assays = ["assay_b", "assay_d"]
    test_assay = ["assay_e", "assay_f"]
    return {"train": train_assays, "test": test_assay}


class TestDistanceUtils:
    def test_create_distance_dataframe(self):
        distance_mat = torch.tensor([1.0, 0.5, 3.5])
        distance_mat_with_nan = torch.tensor([1.0, 0.5, float("nan")])
        assays = ["assay_a", "assay_b", "assay_c"]
        df = create_distance_dataframe(distance_mat, assays)
        df_with_nan = create_distance_dataframe(distance_mat_with_nan, assays)

        assert df.shape == (3, 1)
        assert df.index.tolist() == ["assay_a", "assay_b", "assay_c"]
        assert df_with_nan.shape == (3, 1)

    def test_select_training_assays(self, distance_dict, lookup_table):
        d, assays = select_training_assays(lookup_table, distance_dict)
        assert isinstance(d, torch.Tensor)
        assert isinstance(assays, list)
        assert d.shape == (2, 2)
        assert assays == ["assay_b", "assay_d"]
        assert d.shape[0] == len(assays)
        assert d.tolist() == [[2, 3], [6, 7]]

    def test_select_test_assay(self, distance_dict):
        x = select_test_assay("assay_e", distance_dict["distance_matrices"], distance_dict)
        assert isinstance(x, torch.Tensor)
        assert x.shape == (4,)
        assert x.tolist() == [0, 2, 4, 6]

        with pytest.raises(KeyError):
            x = select_test_assay("assay_h", distance_dict["distance_matrices"], distance_dict)
