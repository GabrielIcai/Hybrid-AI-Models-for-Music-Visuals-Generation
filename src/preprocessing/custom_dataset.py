import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


# Definir las transformaciones
def c_transform(mean, std):
    return transforms.Compose(
        [
            transforms.Resize((155, 155)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# Custom Dataset se encarga de tranformar cada elemento individual del dataset a tensor.
##DEVUELVE:
# image: Tensor de forma (canales, altura, anchura)
# additional_features: Tensor de forma (num_features,)
# labels: Tensor de forma (num_labels,)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, base_path, transform):
        self.data = data
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.base_path, row["Ruta"])

        try:
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            required_columns = [
                "RMS",
                "ZCR",
                "Mean Absolute Amplitude",
                "Crest Factor",
                "Standard Deviation of Amplitude",
                "Spectral Centroid",
                "Spectral Bandwidth",
                "Spectral Roll-off",
                "Spectral Flux",
                "VAD",
                "Spectral Variation",
                "Tempo",
            ]
            for col in required_columns:
                if col not in row:
                    raise ValueError(f"Columna {col} no encontrada en el DataFrame.")

            additional_features = row[required_columns].values.astype(float)
            additional_features = torch.tensor(additional_features, dtype=torch.float32)

            labels = torch.tensor(
                row.iloc[list(range(1, 5)) + list(range(6, 8))].values.astype(float),
                dtype=torch.float32,
            )
            return image, additional_features, labels

        except Exception as e:
            raise RuntimeError(
                f"Error procesando el índice {idx}, archivo {img_path}: {e}"
            )
