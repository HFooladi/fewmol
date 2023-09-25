import argparse
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

CHECKOUT_PATH = Path(__file__).resolve().parent.parent
os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)

from sklearn.model_selection import KFold
from torch.utils.data import random_split

from fewmol.data.dataset import FSMOLDataset
from fewmol.data.evaluate import Evaluator
from fewmol.model import GNN, reset_weights
from fewmol.utils.io_utils import SaveBestModel, save_model, save_plots
from fewmol.utils.train_utils import eval, train

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--gnn",
        type=str,
        default="gin",
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin)",
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--num_layer", type=int, default=3, help="number of GNN message passing layers (default: 3)"
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=100,
        help="dimensionality of hidden units in GNNs (default: 100)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="input batch size for training (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train (default: 100)"
    )
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers (default: 2)")
    parser.add_argument(
        "--dataset", type=str, default="fsmol", help="dataset name (default: fsmol)"
    )
    parser.add_argument(
        "--chembl_id", type=str, default="", help="chembl id for the dataset (default: )"
    )
    parser.add_argument(
        "--k_folds", type=int, default="5", help="number of k_folds cross validation (default: 5)"
    )
    parser.add_argument(
        "--feature", type=str, default="full", help="full feature or simple feature"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="config",
        help='filename to output result (default:"config")',
    )
    parser.add_argument(
        "--train_assay_ids", type=str, default="", help='name of train assay ids (default:"")'
    )

    args = parser.parse_args()
    return args


def main(chembl_id, train_assay_ids):
    args = parse_args()
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    torch.manual_seed(42)

    if args.chembl_id == "":
        print("Please provide a valid chembl id")
        args.chembl_id = chembl_id

    if args.train_assay_ids == "":
        print("Please provide a valid train_assay_ids")
        args.train_assay_ids = train_assay_ids

    ### automatic dataloading and splitting
    dataset = FSMOLDataset(name=args.dataset, chembl_id=args.chembl_id)
    print(dataset)

    if args.feature == "full":
        pass
    elif args.feature == "simple":
        print("using simple feature")
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = {}

    """
    train_idx, test_idx, valid_idx = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    split_idx["train"] = train_idx.indices
    split_idx["valid"] = valid_idx.indices
    split_idx["test"] = test_idx.indices
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    """

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=args.k_folds, shuffle=True)

    output_dir = osp.join("outputs", "epoch10")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start print
    print("--------------------------------")

    test_task_dir = osp.join(output_dir, args.chembl_id)

    results_folds = []
    results = {}

    # K-fold Cross Validation model evaluation
    for fold in range(args.k_folds):
        configs = torch.load(osp.join(test_task_dir, f"config_{fold}.pth"))
        # Define data loaders for training and testing data in this fold

        # Print
        print(f"FOLD {fold}")
        print("--------------------------------")

        split_idx["train"] = configs["train"]
        split_idx["test"] = configs["test"]

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(
            dataset[split_idx["train"]], batch_size=args.batch_size, num_workers=args.num_workers
        )
        valid_loader = DataLoader(
            dataset[split_idx["test"]], batch_size=args.batch_size, num_workers=args.num_workers
        )

        ### automatic evaluator. takes dataset name as input
        evaluator = Evaluator(args.dataset)
        internal_dict = {}

        for train_assay_id in train_assay_ids:
            train_task_dir = osp.join(output_dir, train_assay_id)
            if args.gnn == "gin":
                model = GNN(
                    gnn_type="gin",
                    num_tasks=dataset.num_tasks,
                    num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio,
                    virtual_node=False,
                ).to(device)
            elif args.gnn == "gin-virtual":
                model = GNN(
                    gnn_type="gin",
                    num_tasks=dataset.num_tasks,
                    num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio,
                    virtual_node=True,
                ).to(device)
            elif args.gnn == "gcn":
                model = GNN(
                    gnn_type="gcn",
                    num_tasks=dataset.num_tasks,
                    num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio,
                    virtual_node=False,
                ).to(device)
            elif args.gnn == "gcn-virtual":
                model = GNN(
                    gnn_type="gcn",
                    num_tasks=dataset.num_tasks,
                    num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio,
                    virtual_node=True,
                ).to(device)
            else:
                raise ValueError("Invalid GNN type")

            for train_fold in range(1):
                checkpoint = osp.join(
                    train_task_dir, f"{train_assay_id}_{train_fold}_final_model.pth"
                )
                model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
                model.to(device)

                optimizer = optim.Adam(model.parameters(), lr=0.0001)

                valid_curve = []
                # test_curve = []
                train_curve = []

                for epoch in range(1, args.epochs + 1):
                    print("=====Epoch {}".format(epoch))
                    print("Training...")
                    train(model, device, train_loader, optimizer, dataset.task_type)

                    print("Evaluating...")
                    train_perf = eval(model, device, train_loader, evaluator)
                    valid_perf = eval(model, device, valid_loader, evaluator)
                    # test_perf = eval(model, device, test_loader, evaluator)

                    print({"Train": train_perf, "Validation": valid_perf, "Test": valid_perf})

                    train_curve.append(train_perf[dataset.eval_metric])
                    valid_curve.append(valid_perf[dataset.eval_metric])
                    # test_curve.append(test_perf[dataset.eval_metric])

                if "classification" in dataset.task_type:
                    best_val_epoch = np.argmax(np.array(valid_curve))
                    best_train = max(train_curve)
                else:
                    best_val_epoch = np.argmin(np.array(valid_curve))
                    best_train = min(train_curve)

                results_folds.append(valid_curve[-1])
                print(f"Finished training for fold {fold}!")
                print("Best validation score: {}".format(valid_curve[best_val_epoch]))
                print("Test score: {}".format(valid_curve[best_val_epoch]))

                # Internal_dict
                internal_dict.update({f"{train_assay_id}_{train_fold}": valid_curve[-1]})

        results[f"{args.chembl_id}_{fold}"] = internal_dict

    with open(f"{test_task_dir}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    with open("datasets/fsmol/fsmol-0.2.json", "r") as f:
        assay_ids = json.load(f)

    test_assay_ids = assay_ids["test"]
    train_assay_ids = assay_ids["train"]

    for chembl_id in tqdm(test_assay_ids):
        if osp.exists(osp.join("outputs", "epoch10", chembl_id + ".json")):
            continue
        print(f"Running {chembl_id}")
        main(chembl_id, train_assay_ids)
        print(f"Finished {chembl_id}")
