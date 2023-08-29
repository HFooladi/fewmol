import os
import os.path as osp
import sys
import json
from pathlib import Path

from tqdm import tqdm
import argparse
import time
import numpy as np

import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict, List

CHECKOUT_PATH = Path(__file__).resolve().parent.parent
os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)

from fewmol.model.gnn import GNN, reset_weights
from torch.utils.data import random_split
from sklearn.model_selection import KFold

from fewmol.data.dataset import FSMOLDataset
from fewmol.data.evaluate import Evaluator
from fewmol.utils.io_utils import SaveBestModel, save_model, save_plots
from fewmol.utils.train_utils import train, eval


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

    args = parser.parse_args()
    return args


def main(chembl_id):
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

    output_dir = osp.join("outputs", "finetuning", args.chembl_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start print
    print("--------------------------------")

    results_folds = []
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        # save_best_model = SaveBestModel(name=f'{args.chembl_id}_{fold}')

        # Print
        print(f"FOLD {fold}")
        print("--------------------------------")

        split_idx["train"] = train_ids
        split_idx["test"] = valid_ids

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_subsampler,
            num_workers=args.num_workers,
        )
        valid_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=valid_subsampler,
            num_workers=args.num_workers,
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

        model.apply(reset_weights)

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

            # save_best_model(valid_curve[-1], epoch, model, optimizer, criterion=cls_criterion)

        # save the best model for this fold
        save_model(
            args.epochs,
            model,
            optimizer,
            criterion=cls_criterion,
            output_dir=output_dir,
            name=f"{args.chembl_id}_{fold}",
        )

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

        if not args.filename == "":
            result = {}
            result.update({"Val": valid_curve, "Train": train_curve})
            result.update(split_idx)
            configs = vars(args)
            result.update(configs)
            torch.save(result, osp.join(output_dir, f"{args.filename}_{fold}.pth"))

    average = np.average(results_folds)
    std = np.std(results_folds)
    print(f"Average: {average}")
    print(f"Std: {std}")
    if not args.filename == "":
        torch.save({"average": average, "std": std}, osp.join(output_dir, f"{args.filename}.pth"))


if __name__ == "__main__":
    with open("datasets/fsmol/fsmol-0.1.json", "r") as f:
        assay_ids = json.load(f)

    test_assay_ids = assay_ids["test"]

    for chembl_id in test_assay_ids:
        print(f"Running {chembl_id}")
        main(chembl_id)
        print(f"Finished {chembl_id}")
