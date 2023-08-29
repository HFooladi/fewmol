import torch
from fewmol.utils.train_utils import eval


# test the last epoch saved model
def test_last_model(model, checkpoint, test_loader):
    print("Loading last epoch saved model weights...")
    assert "model_state_dict" in checkpoint.keys(), "No model weights found in checkpoint!"
    model.load_state_dict(checkpoint["model_state_dict"])
    test_acc = eval(model, test_loader)
    print(f"Last epoch saved model accuracy: {test_acc:.3f}")


# test the best epoch saved model
def test_best_model(model, checkpoint, test_loader):
    print("Loading best epoch saved model weights...")
    assert "model_state_dict" in checkpoint.keys(), "No model weights found in checkpoint!"
    model.load_state_dict(checkpoint["model_state_dict"])
    test_acc = eval(model, test_loader)
    print(f"Best epoch saved model accuracy: {test_acc:.3f}")
