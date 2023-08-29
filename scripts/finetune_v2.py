import os
import os.path as osp
import sys
import json

from tqdm import tqdm
import argparse
import time
import numpy as np
import pickle
import pandas as pd
from typing import List, Dict

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import torch.optim as optim
import torch.nn.functional as F

CHECKOUT_PATH = os.path.join(os.environ["HOME"], "Documents", "hfooladi", "TLMOL", "fewmol")
os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)

from model.gnn import GNN, reset_weights
from torch.utils.data import random_split
from sklearn.model_selection import KFold

from fewmol.data.dataset import FSMOLDataset
from fewmol.data.evaluate import Evaluator
from fewmol.utils.io_utils import SaveBestModel, save_model, save_plots
from fewmol.utils.distance_utils import (
    select_training_assays,
    select_test_assay,
    create_distance_dataframe,
    normalize,
)


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(
                    pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
                )
            else:
                loss = reg_criterion(
                    pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
                )
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator) -> Dict:
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


"""
def parse_opts():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 3)')
    parser.add_argument('--emb_dim', type=int, default=100,
                        help='dimensionality of hidden units in GNNs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers (default: 2)')
    parser.add_argument('--dataset', type=str, default="fsmol",
                        help='dataset name (default: fsmol)')
    parser.add_argument('--chembl_id', type=str, default="",
                        help='chembl id for the dataset (default: )')
    parser.add_argument('--k_folds', type=int, default="5",
                        help='number of k_folds cross validation (default: 5)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="config",
                        help='filename to output result (default:"config")')
    parser.add_argument('--train_assay_ids', type=str, default="",
                        help='name of train assay ids (default:"")')
    parser.add_argument('--k_nearest', type=int, default="5",
                    help='number of k nearest dataset to consider (default: 5)')
    parser.add_argument('--strategy', type=str, default="selective", choices=['selective', 'random'],
                        help='distance type (default: chemical)')
    

    args = parser.parse_args()
    return args
"""


def main(chembl_id):
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
    parser.add_argument(
        "--k_nearest",
        type=int,
        default="5",
        help="number of k nearest dataset to consider (default: 5)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="selective",
        choices=["selective", "random"],
        help="distance type (default: chemical)",
    )
    args = parser.parse_args()

    print("--------- START ---------")
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    torch.manual_seed(42)

    if args.chembl_id == "":
        print("Please provide a valid chembl id")
        args.chembl_id = chembl_id

    train_assay_path = "./datasets/fsmol/fsmol-0.2.json"
    with open(train_assay_path, "r") as f:
        fsmol_lookup = json.load(f)

    chemical_hardness_path = (
        "./datasets/fsmol/distance/inter/fsmol_distance_matrices_gin_supervised_masking.pkl"
    )
    with open(chemical_hardness_path, "rb") as f:
        fsmol_chemical_distance = pickle.load(f)

    protein_hardness_path = "./datasets/fsmol/distance/protein/protein_distance_matrix.pkl"
    with open(protein_hardness_path, "rb") as f:
        fsmol_protein_distance = pickle.load(f)

    d_mat, train_assays = select_training_assays(fsmol_lookup, fsmol_chemical_distance)
    d_mat_protein, train_assays_protein = select_training_assays(
        fsmol_lookup, fsmol_protein_distance
    )

    chemical_df = create_distance_dataframe(d_mat, train_assays)
    protein_df = create_distance_dataframe(d_mat_protein, train_assays_protein)

    chemical_df["chemical_distance"] = normalize(chemical_df["chemical_distance"])
    protein_df["protein_distance"] = normalize(protein_df["protein_distance"])

    dist_df = pd.merge(chemical_df, protein_df, left_index=True, right_index=True)
    dist_df["distance"] = dist_df["chemical_distance"] + dist_df["protein_distance"]

    if args.strategy == "selective":
        final_df = dist_df.sort_values(by="distance", ascending=True).head(args.k_nearest)
        train_assay_ids = final_df.index.tolist()
    elif args.strategy == "random":
        final_df = dist_df.sample(n=args.k_nearest)
        train_assay_ids = final_df.index.tolist()

    training_tasks = []
    for assay_id in train_assay_ids:
        data = FSMOLDataset(name=args.dataset, chembl_id=assay_id)
        training_tasks.extend([*data])

    training_dataset = Batch(training_tasks)
    print(training_dataset)

    train_size = int(0.8 * len(training_dataset))
    test_size = len(training_dataset) - train_size
    train_dataset, valid_dataset = random_split(
        training_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

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

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_curve = []
    valid_curve = []

    save_best_model = SaveBestModel(name=f"{args.chembl_id}")

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, optimizer, dataset.task_type)

        print("Evaluating...")
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)

        print({"Train": train_perf, "Validation": valid_perf, "Test": valid_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])

        save_best_model(valid_curve[-1], epoch, model, optimizer, criterion=cls_criterion)

    # save the best model for this fold
    save_model(
        args.epochs,
        model,
        optimizer,
        criterion=cls_criterion,
        output_dir="outputs/finetuning",
        name=f"{args.chembl_id}",
    )

    if "classification" in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print("------------------------------------------")
    print("----------FINETUNING ON FSMOL-------------")
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

    test_task_dir = osp.join("outputs/epoch10", args.chembl_id)
    # Define data loaders for training and testing data in this fold
    configs = torch.load(osp.join(test_task_dir, f"config_3.pth"))
    # Define data loaders for training and testing data in this fold

    # Print
    print("--------------------------------")
    split_idx = {}
    split_idx["train"] = configs["train"]
    split_idx["test"] = configs["test"]

    # Define data loaders for training and testing data in this fold
    test_task_train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_task_test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=args.batch_size, num_workers=args.num_workers
    )

    for epoch in range(1, 11):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train(model, device, test_task_train_loader, optimizer, dataset.task_type)

        print("Evaluating...")
        train_perf = eval(model, device, test_task_train_loader, evaluator)
        valid_perf = eval(model, device, test_task_test_loader, evaluator)

        print({"Train": train_perf, "Validation": valid_perf, "Test": valid_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])


if __name__ == "__main__":
    print(torch.__version__)
    with open("datasets/fsmol/fsmol-0.2.json", "r") as f:
        assay_ids = json.load(f)

    chembl_id = assay_ids["test"][0]
    main(chembl_id)
