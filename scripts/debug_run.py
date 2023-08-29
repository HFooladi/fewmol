import os
import os.path as osp
import sys
import json

from tqdm import tqdm
import argparse
import time
import numpy as np

import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F

CHECKOUT_PATH = os.path.join(os.environ["HOME"], "Documents", "hfooladi", "TLMOL", "fewmol")
os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)

from model.gnn import GNN, reset_weights
from torch.utils.data import random_split
from sklearn.model_selection import KFold

from data.dataset import FSMOLDataset
from data.evaluate import Evaluator
from utils.io_utils import SaveBestModel, save_model, save_plots


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


def eval(model, device, loader, evaluator):
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


def parser():
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


def main():
    args = parser()
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
