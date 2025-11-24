import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

from src.data import get_dataloaders
from src.utils import save_model, set_seed
from src.models import BreastMLPClassifier
from src.train_functions import train_step, val_step, test_step
from src.eval import evaluate_model

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

import joblib
import mlflow
import mlflow.sklearn
import json

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Breast MLP Classifier")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[128, 256], help="Hidden layer dimensions")
    parser.add_argument("--info_filename", type=str, default=r"data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx")
    parser.add_argument("--images_folder", type=str, default=r"data/BrEaST-Lesions_USG-images_and_masks/")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Create TensorBoard run
    name = f"Tab_model_lr_{args.lr}_bs_{args.batch_size}_hd_{str(args.hidden_dims)}_dropout_{args.dropout}"
    writer = SummaryWriter(f"runs/{name}")

    with mlflow.start_run():
        # Load segmentation data
        train_loader, val_loader, test_loader = get_dataloaders(
            info_file=args.info_filename,
            images_folder=args.images_folder,
            batch_size=args.batch_size,
            type="tabular"
        )

        # Model
        x0, _ = next(iter(train_loader))
        input_dim = x0.size(1)
        model = BreastMLPClassifier(
            input_dim=input_dim,
            num_classes=3,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout
        ).to(device)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_trainable:,}")

        # Optimizer & loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        # Train loop
        with tqdm(range(args.epochs)) as pbar:
            for epoch in pbar:
                train_step(model, train_loader, loss_fn, optimizer, device, writer, epoch)
                acc, val_loss = val_step(model, val_loader, loss_fn, device, writer, epoch)
                pbar.set_description(f"Epoch {epoch} | Val Accuracy: {acc:.4f} | Val Loss: {val_loss:.4f}")

        # Save model
        joblib.dump(model, "model.pkl")

        mlflow.sklearn.log_model(model, "classification-model")
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("hidden_dims", args.hidden_dims)
        mlflow.log_param("epochs", args.epochs)

        # Test
        test_acc = test_step(model, test_loader, device)
        mlflow.log_metric("test_acc", test_acc)

        evaluate_model(model, test_loader, device)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        metrics = {
            "accuracy": test_acc
            }
        with open("mlflow_metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    main()
