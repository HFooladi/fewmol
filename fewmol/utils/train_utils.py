import torch

from tqdm import tqdm
from typing import Dict, List

from fewmol.data import Evaluator


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train_one_epoch(model, device, loader, optimizer, task_type):
    """
    Train the model for one epoch.
    """
    running_loss = 0.0
    model.train()
    counter = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        counter += 1
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
            running_loss += loss.item()
    return running_loss / counter


def validation_one_epoch(model, device, loader, task_type):
    """
    validation loss of the model for one epoch.
    """
    running_vloss = 0.0
    model.eval()

    counter = 0 
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            counter += 1
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                pred = model(batch)
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
                running_vloss += loss.item()
    return running_vloss / counter


def eval(model, device, loader, evaluator) -> Dict:
    """
    Evaluate the model on validation / test set.
    """
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
