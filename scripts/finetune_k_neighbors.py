"""
This script is used to finetune the model on the each test tasks. First for each test task,
We are finding k-nearest neighbors (training datasets) based on the distance matrix. Then, we trained
the model on the k-nearest neighbors and finetune the model on the test task.
For each test task, we are using predefined split for training and validation.
"""

import argparse
import json
import os
import os.path as osp
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

CHECKOUT_PATH = Path(__file__).resolve().parent.parent
os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)


from fewmol.data import Evaluator, FSMOLDataset
from fewmol.model import GNN, reset_weights
from fewmol.utils import SaveBestModel, final_distance_df, save_model, save_plots
from fewmol.utils.train_utils import eval, train_one_epoch, validation_one_epoch

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

TRAIN_ASSAY_PATH = "./datasets/fsmol/fsmol-0.3.json"
CHEMICAL_HARDNESS_PATH = (
    "./datasets/fsmol/distance/inter/fsmol_distance_matrices_gin_supervised_masking.pkl"
)
PROTEIN_HARDNESS_PATH = "./datasets/fsmol/distance/protein/protein_distance_matrix.pkl"

FOLD_TO_SPLIT = {0: 8, 1: 16, 2: 32, 3: 64, 4: 100}
SPLIT_TO_FOLD = {8: 0, 16: 1, 32: 2, 64: 3, 100: 4}


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
    parser.add_argument(
        "--finetuning_epochs", type=int, default=40, help="number of epochs to train (default: 10)"
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
        "--split_size",
        type=int,
        default="64",
        help="number of requested training points for finetuning (default: 64)",
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
        default="10",
        help="number of k nearest dataset to consider (default: 10)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="selective",
        choices=["selective", "random"],
        help="distance type (default: chemical)",
    )

    args = parser.parse_args()
    return args


def main(chembl_id):
    args = parse_args()
    fold = SPLIT_TO_FOLD[args.split_size]
    dataset = FSMOLDataset(name=args.dataset, chembl_id=args.chembl_id)

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
    
    output_dir = osp.join("outputs", "finetuning", args.chembl_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(logdir=f"outputs/finetuning/runs/{args.gnn}/{args.chembl_id}", comment=args.gnn)

    with open(TRAIN_ASSAY_PATH, "r") as f:
        fsmol_lookup = json.load(f)

    with open(CHEMICAL_HARDNESS_PATH, "rb") as f:
        fsmol_chemical_distance = pickle.load(f)

    with open(PROTEIN_HARDNESS_PATH, "rb") as f:
        fsmol_protein_distance = pickle.load(f)

    dist_df = final_distance_df(
        fsmol_lookup,
        fsmol_chemical_distance,
        fsmol_protein_distance,
        args.chembl_id,
        operation="sum",
    )

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

    training_dataset = Batch.from_data_list(training_tasks)
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

    save_best_model = SaveBestModel(name=f"{args.chembl_id}", fold=f"{SPLIT_TO_FOLD[args.split_size]}")

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_loss = train_one_epoch(model, device, train_loader, optimizer, dataset.task_type)
        validation_loss = validation_one_epoch(model, device, valid_loader, dataset.task_type)

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
        output_dir=output_dir,
        name=f"{args.chembl_id}",
    )

    if "classification" in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print("-----------------------------------------------------------")
    print("-------------------FINETUNING ON FSMOL---------------------")

    ### automatic dataloading and splitting
    print(dataset)
    if args.feature == "full":
        pass
    elif args.feature == "simple":
        print("using simple feature")
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    test_task_dir = osp.join("outputs/training/gin", args.chembl_id)
    # Define data loaders for training and testing data in this fold
    configs = torch.load(osp.join(test_task_dir, f"config_{fold}.pth"))
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

    for epoch in range(1, args.finetuning_epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Finetuning...")
        test_train_loss = train_one_epoch(
            model, device, test_task_train_loader, optimizer, dataset.task_type
        )
        test_validation_loss = validation_one_epoch(
            model, device, test_task_test_loader, dataset.task_type
        )

        print("Evaluating...")
        test_train_perf = eval(model, device, test_task_train_loader, evaluator)
        test_valid_perf = eval(model, device, test_task_test_loader, evaluator)

        print(
            {
                "Train": test_train_perf[dataset.eval_metric],
                "Validation": test_valid_perf[dataset.eval_metric],
                "Test": test_valid_perf[dataset.eval_metric],
            }
        )

        train_curve.append(test_train_perf[dataset.eval_metric])
        valid_curve.append(test_valid_perf[dataset.eval_metric])

        writer.add_scalar(
            f"{args.chembl_id}_data_{SPLIT_TO_FOLD[args.split_size]}/Finetuned_Train_Loss",
            test_train_loss,
            epoch,
        )
        writer.add_scalar(
            f"{args.chembl_id}_data_{SPLIT_TO_FOLD[args.split_size]}/Finetuned_Validation_Loss",
            test_validation_loss,
            epoch,
        )
        writer.add_scalar(
            f"{args.chembl_id}_data_{SPLIT_TO_FOLD[args.split_size]}/Finetuned_Train_acc",
            train_perf[dataset.eval_metric],
            epoch,
        )
        writer.add_scalar(
            f"{args.chembl_id}_data_{SPLIT_TO_FOLD[args.split_size]}/Finetuned_Validation_acc",
            valid_perf[dataset.eval_metric],
            epoch,
        )
        writer.add_scalars(
            f"{args.chembl_id}_data_{SPLIT_TO_FOLD[args.split_size]}/Finetuned_Training vs. Validation Metric",
            {
                "Training": test_train_perf[dataset.eval_metric],
                "Validation": test_valid_perf[dataset.eval_metric],
            },
            epoch,
        )
        writer.add_scalars(
            f"{args.chembl_id}_data_{SPLIT_TO_FOLD[args.split_size]}/Finetuned_Training vs. Validation Loss",
            {
                "Training": test_train_loss,
                "Validation": test_validation_loss,
            },
            epoch,
        )


if __name__ == "__main__":
    print(torch.__version__)
    with open("datasets/fsmol/fsmol-0.3.json", "r") as f:
        assay_ids = json.load(f)

    chembl_id = assay_ids["test"][0]
    main(chembl_id)
