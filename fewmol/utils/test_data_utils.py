import pytest
from fewmol.data import FSMOLDataset
from fewmol.utils import get_split, get_split_multiple_size, get_split_multiple_assay


@pytest.fixture
def chembl_ids():
    return ["CHEMBL1119333", "CHEMBL1243967", "CHEMBL1243970", "CHEMBL1614292", "CHEMBL1614433"]


@pytest.fixture
def training_size_list():
    return [8, 16, 32]


class TestDataUtils:
    def test_get_split(self, chembl_ids):
        chembl_id = chembl_ids[0]
        split = get_split(chembl_id, requested_training_size=16)
        assert len(split[chembl_id]["16"]["train"]) == 16
        assert list(split.keys()) == [chembl_id]
        assert list(split[chembl_id].keys()) == ["16"]
        assert isinstance(split[chembl_id]["16"]["train"], list)
        assert isinstance(split[chembl_id]["16"]["test"], list)

    def test_get_split_multiple_size(self, chembl_ids, training_size_list):
        chembl_id = chembl_ids[0]
        split = get_split_multiple_size(chembl_id, training_size_list)
        assert len(split[chembl_id]) == len(training_size_list)
        assert list(split.keys()) == [chembl_id]
        assert list(split[chembl_id].keys()) == [str(size) for size in training_size_list]
        assert isinstance(split[chembl_id]["8"]["train"], list)
        assert isinstance(split[chembl_id]["8"]["test"], list)

        for size in training_size_list:
            assert len(split[chembl_id][str(size)]["train"]) == size
        
