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
import matplotlib as plt 
import os

import joblib
import mlflow
import mlflow.sklearn


tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

def main():
    epochs = 50
    lr = 5e-4
    batch_size = 8
    dropout = 0.5
    hidden_dims = (128, 256)

    info_filename = r"data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    images_folder = r"data/BrEaST-Lesions_USG-images_and_masks/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Create TensorBoard run
    name = f"Tab_model_lr_{lr}_bs_{batch_size}_hd_{str(hidden_dims)}_dropout_{dropout}"
    writer = SummaryWriter(f"runs/{name}")

    with mlflow.start_run():
        # Load segmentation data
        train_loader, val_loader, test_loader = get_dataloaders(
            info_file=info_filename,
            images_folder=images_folder,
            batch_size=batch_size,
            type="tabular"
        )

        # Model
        x0, _ = next(iter(train_loader))
        input_dim = x0.size(1)
        model = BreastMLPClassifier(
            input_dim=input_dim,
            num_classes=3,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(device)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_trainable:,}")

        # Optimizer & loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # Train loop
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                train_step(model, train_loader, loss_fn, optimizer, device, writer, epoch)

                acc, val_loss = val_step(
                    model, val_loader, loss_fn, device, writer, epoch
                )

                pbar.set_description(f"Epoch {epoch} | Val Accuracy: {acc:.4f} | Val Loss: {val_loss:.4f}")

        # Save model

        joblib.dump(model, "model.pkl")

        mlflow.sklearn.log_model(model, "classification-model")
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("hidden_dims", hidden_dims)
        mlflow.log_param("epochs", epochs)

        # save_model(model, name)

        # Test
        test_acc = test_step(model, test_loader, device)
        mlflow.log_metric("test_acc", test_acc)

        evaluate_model(model, test_loader, device)
        print(f"\nTest Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()