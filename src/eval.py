import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import accuracy, set_seed, load_model
from src.data import get_dataloaders
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.makedirs("./images", exist_ok=True)


def evaluate_model(model, dataloader, device, image_name: str = "confusion_matrix"):
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_softmax = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(outputs)
            all_softmax.append(probs)

            all_targets.append(labels)

    # Concatenar todos los batchs
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_softmax = torch.cat(all_softmax)

    # Calcular métricas
    acc = accuracy(all_preds, all_targets)


    print(f"Accuracy: {acc*100:.2f}%")


    # Matriz de confusión
    preds_labels = torch.argmax(all_preds, dim=1).cpu().numpy()
    true_labels = all_targets.cpu().numpy()
    cm = confusion_matrix(true_labels, preds_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["benign", "malignant", "normal"],
        yticklabels=["benign", "malignant", "normal"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"{image_name}.png")

    return preds_labels, true_labels, all_softmax


def main():
    model_name = "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2"
    batch_size = 32
    seed = 42
    info_filename: str = r"data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    images_and_masks_foldername: str = r"data/BrEaST-Lesions_USG-images_and_masks/"

    set_seed(seed)

    _, _, test_data = get_dataloaders(
        info_filename, images_and_masks_foldername, batch_size, seed=seed, type="tabular"
    )

    model = load_model(model_name)

    _, _, _ = evaluate_model(
        model, test_data, device, image_name=f"{model_name}_confusion_matrix"
    )


if __name__ == "__main__":
    main()
