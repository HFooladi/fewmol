import os
import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, download_url
import pickle
from fewmol.data import create_pyg_graph_from_smiles


class FSMOLDataset(InMemoryDataset):
    def __init__(
        self,
        chembl_id,
        name="fsmol",
        root="datasets",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        meta_dict=None,
    ):
        """
        - name (str): name of the dataset
        - chembl_id (str): chembl id of the dataset
        - root (str): root directory to store the dataset folder
        - transform, pre_transform (optional): transform/pre-transform graph objects
        - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                        but when something is passed, it uses its information. Useful for debugging for external contributers.
        """
        self.name = name
        self.chembl_id = chembl_id

        if meta_dict is None:
            self.dir_name = "_".join(name.split("-"))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + "_pyg")):
                self.dir_name = self.dir_name + "_pyg"

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(
                os.path.join(os.path.dirname(__file__), "master.csv"),
                index_col=0,
                keep_default_na=False,
            )
            if not self.name in master:
                error_mssg = "Invalid dataset name {}.\n".format(self.name)
                error_mssg += "Available datasets are as follows:\n"
                error_mssg += "\n".join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict["dir_path"]
            self.original_root = ""
            self.root = meta_dict["dir_path"]
            self.meta_info = meta_dict

        self.num_tasks = int(self.meta_info["num tasks"])
        self.eval_metric = self.meta_info["eval metric"]
        self.task_type = self.meta_info["task type"]
        self.__num_classes__ = int(self.meta_info["num classes"])
        self.binary = self.meta_info["binary"] == "True"

        super(FSMOLDataset, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["fsmol_test.pkl", "fsmol_train.pkl"]

    @property
    def processed_file_names(self):
        return [f"{self.chembl_id}.pt"]

    def process(self):
        # Read data into huge `Data` list.
        with open(self.raw_paths[0], "rb") as f:
            fsmol_data = pickle.load(f)

        add_inverse_edge = self.meta_info["add_inverse_edge"] == "True"

        if self.meta_info["additional node files"] == "None":
            additional_node_files = []
        else:
            additional_node_files = self.meta_info["additional node files"].split(",")

        if self.meta_info["additional edge files"] == "None":
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info["additional edge files"].split(",")

        # Process to PyG data object
        smiles = fsmol_data[self.chembl_id]["smiles"]
        labels = fsmol_data[self.chembl_id]["labels"]
        data_list = create_pyg_graph_from_smiles(
            smiles_string=smiles,
            labels=labels,
            additional_node_files=additional_node_files,
            additional_edge_files=additional_edge_files,
        )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_fsmol_dataset(
    name="fsmol", transform=None, pre_transform=None, pre_filter=None, meta_dict=None
):
    """
    Create dataset object for all the chembl ids in the dataset

    - name (str): name of the dataset
    - transform, pre_transform (optional): transform/pre-transform graph objects
    - meta_dict: dictionary that stores all the meta-information about data. Default is None,
    """
    with open("dataset/fsmol/raw/fsmol_test.pkl", "rb") as f:
        fsmol_data = pickle.load(f)
    for key, value in fsmol_data.items():
        chembl_id = key
        dataset = FSMOLDataset(
            chembl_id,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            meta_dict=meta_dict,
        )
    return dataset
