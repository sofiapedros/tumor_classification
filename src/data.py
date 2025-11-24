import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, random_split

from PIL import Image
import pandas as pd
import numpy as np
import os


class DatasetBaseBreast:
    def __init__(self, info_filename: str, images_and_masks_foldername: str):
        """
        Args:
            info_filename: path to the Excel file with all the information
            images_and_masks_foldername: path to images and masks
        """

        df = pd.read_excel(info_filename, sheet_name="BrEaST-Lesions-USG clinical dat")

        self.df = df.reset_index(drop=True)

        self.images_folder = images_and_masks_foldername

        # --- Mapeo de etiquetas para clasificación ---
        self.labels = (
            df["Classification"]
            .map({"benign": 0, "malignant": 1, "normal": 2})
            .astype(int)
            .tolist()
        )

        # --- Rutas a imágenes ---
        self.image_paths = [
            os.path.join(images_and_masks_foldername, img) for img in df["Image_filename"]
        ]

        # --- Rutas a máscaras, si existen ---
        self.mask_paths = [
            (
                os.path.join(images_and_masks_foldername, str(m))
                if isinstance(m, str)
                else None
            )
            for m in df["Mask_tumor_filename"]
        ]

        # --- Datos tabulares (excepto columnas no numéricas hardcore) ---
        self.tabular_df = df.copy()

        # Elimina nombres de archivos, IDs y otros que no quieres como input
        drop_cols = [
            "CaseID",
            "Image_filename",
            "Mask_tumor_filename",
            "Mask_other_filename",
            "Diagnosis",
            "Verification",
            "Interpretation",
            "BIRADS",
            "Classification"
        ]
        self.tabular_df = self.tabular_df.drop(columns=drop_cols, errors="ignore")

        def map_yes_no_na(col):
            if set(col.dropna().unique()).issubset({"yes", "no", "not applicable"}):
                return col.map({"yes": 1, "no": 0, "not applicable": 2}).astype(np.float32)
            return col

        self.tabular_df = self.tabular_df.apply(map_yes_no_na)
    
        # Para el resto de columnas categóricas, aplicar one-hot
        categorical_cols = self.tabular_df.select_dtypes(include="object").columns
        self.tabular_df = pd.get_dummies(self.tabular_df, columns=categorical_cols)

        # Convertir todo a float32
        for col in self.tabular_df.columns:
            self.tabular_df[col] = pd.to_numeric(self.tabular_df[col], errors="coerce").fillna(0).astype(np.float32)

        # Normalizar solo columnas numéricas que no sean de tipo 0/1
        binary_cols = [col for col in self.tabular_df.columns if set(self.tabular_df[col].unique()).issubset({0.0, 1.0})]
        numeric_cols = self.tabular_df.columns.difference(binary_cols)
        self.tabular_df[numeric_cols] = (
            self.tabular_df[numeric_cols] - self.tabular_df[numeric_cols].mean()
        ) / (self.tabular_df[numeric_cols].std() + 1e-6)

    def __len__(self) -> int:
        return len(self.df)

    def load_image(self, idx: int) -> Image:
        """
        Load the corresponding image
        Args:
        - idx (int): index of the image to load
        Returns:
        - Image
        """
        img_path = self.image_paths[idx]
        return Image.open(img_path).convert("RGB")

    def load_mask(self, idx) -> Image:
        """
        Load the corresponding mask
        Args:
        - idx (int): index of the mask to load
        Returns:
        - Image
        """
        mask_path = self.mask_paths[idx]
        if mask_path is None or not os.path.exists(mask_path):
            return None
        return Image.open(mask_path).convert("L")  # grayscale

    def load_tabular(self, idx: int) -> torch.Tensor:
        """
        Load the corresponding tabular data
        Args:
        - idx (int): index of the mask to load
        Returns:
        - torch.Tensor
        """
        row = self.tabular_df.iloc[idx]
        return torch.tensor(row.values, dtype=torch.float32)

    def load_label(self, idx: int):
        """
        Load the corresponding label
        Args:
        - idx (int): index of the label to load
        Returns:
        - torch.Tensor
        """
        return torch.tensor(self.labels[idx], dtype=torch.long)


class DatasetTabular(Dataset):
    def __init__(self, base_dataset: DatasetBaseBreast) -> None:
        """
        Dataset for classification
        Args:
        - baseDataset: complete base Dataset
        """
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple:
        """
        Return the corresponding data and label
        Args:
        - idx (int): index of the pair to load
        Returns:
        - tuple with (data, label)
        """
        x = self.base.load_tabular(idx)
        y = self.base.load_label(idx)
        return x, y


def get_dataloaders(
    info_file: str,
    images_folder: str,
    batch_size: int = 32,
    split_ratio: float = 0.8,
    seed: int = 42,
    type: str = "tabular",
) -> tuple:
    """
    Get the dataloaders for the desired model
    Args:
    - info_filename (str): path to the Excel file with all the information
    - images_folder (str): path to images and masks
    - batch_size (int): batch size
    - split_ratio (float): split ratio between train and val
    - seed (int): seed
    - type (str): type of dataset to load (image, tabular, segmentation)
    Returns:
    - tuple with train_dataloader, val_dataloader, test_dataloader
    """

    base_dataset: DatasetBaseBreast = DatasetBaseBreast(
        info_filename=info_file, images_and_masks_foldername=images_folder
    )

    # Generator with fixed seed
    generator = torch.Generator().manual_seed(seed)

    if type == "image":
        dataset = DatasetImagenClasificacion(base_dataset)
    elif type == "tabular":
        dataset = DatasetTabular(base_dataset)
    elif type == "segmentation":
        dataset = DatasetSegmentacion(base_dataset)
    else:
        raise ValueError(
            "Not a valid value of type. Possible values are: 'image', 'tabular' and 'segmentation'"
        )

    # Split between train, val and test
    n_total = len(base_dataset)
    n_train = int(n_total * split_ratio)
    n_val = int((n_total - n_train) / 2)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    # Get dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader
