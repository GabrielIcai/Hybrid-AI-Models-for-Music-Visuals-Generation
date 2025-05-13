import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd


# Definir las transformaciones
def c_transform(mean, std):
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
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
        self.data = data.reset_index(drop=True)
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Índice {idx} fuera de rango")
        row = self.data.iloc[idx]
        ruta=row["Ruta"]
        img_path = os.path.join(self.base_path, ruta)

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
            #Selecciono 
            label_columns = [
            "Ambient",
            "Deep House",
            "Techno",
            "Trance",
            "Progressive House"]
            labels = torch.tensor(row[label_columns].values.astype(int),dtype=torch.long
        )
            return image, additional_features, labels, img_path

        except Exception as e:
            raise RuntimeError(
                f"Error procesando el índice {idx}, archivo {img_path}: {e}"
            )

import torch
import os
from PIL import Image

class PredictionDatasetGenre(torch.utils.data.Dataset):
    def __init__(self, data, base_path, transform=None):
        self.data = data.reset_index(drop=True)
        self.base_path = base_path
        self.transform = transform

        # Definir columnas de características necesarias
        self.required_columns = [
            "RMS", "ZCR", "Mean Absolute Amplitude", "Crest Factor",
            "Standard Deviation of Amplitude", "Spectral Centroid",
            "Spectral Bandwidth", "Spectral Roll-off", "Spectral Flux",
            "VAD", "Spectral Variation", "Tempo"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Índice {idx} fuera de rango")

        row = self.data.iloc[idx]
        img_path = os.path.join(self.base_path, row["Ruta"])

        try:
            # Cargar imagen
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            # Extraer características adicionales
            additional_features = row[self.required_columns].values.astype(float)
            additional_features = torch.tensor(additional_features, dtype=torch.float32)

            return image, additional_features, img_path

        except Exception as e:
            raise RuntimeError(f"Error procesando el índice {idx}, archivo {img_path}: {e}")

        

##################################################CUSTOM DATASET EMOCIONES##################################################

class EmotionDataset(Dataset):
    def __init__(self, data, base_path, transform=None):
        self.data = data.reset_index(drop=True)
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Índice {idx} fuera de rango")

        row = self.data.iloc[idx]
        img_path = os.path.join(self.base_path, row["Ruta"]) 

        try:
            if not os.path.exists(img_path):
                print(f"Error: La imagen no existe en {img_path}")
                return None, None, None, None

            # Cargar imagen
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            # Características adicionales
            required_features = [
                "RMS", "ZCR", "Crest Factor",
                "Standard Deviation of Amplitude", "Spectral Centroid",
                "Spectral Bandwidth", "Spectral Roll-off", "Spectral Flux",
                "VAD", "Spectral Variation",
            ]
            additional_features = torch.tensor(
                row[required_features].values.astype(float), dtype=torch.float32
            )

            # Etiquetas de valencia y arousal en one-hot encoding 
            valencia_cols = [f"Valencia_{i/10:.1f}" for i in range(11)]
            arousal_cols = [f"Arousal_{i/10:.1f}" for i in range(11)]
            valencia_index = row[valencia_cols].values.argmax()
            arousal_index = row[arousal_cols].values.argmax()
            valencia_value=valencia_index/10
            arousal_value=arousal_index/10
            valencia_label = torch.tensor([valencia_value], dtype=torch.float32)
            arousal_label = torch.tensor([arousal_value], dtype=torch.float32)

            return image, additional_features, valencia_label, arousal_label  

        except Exception as e:
            print(f"Error al cargar la imagen {img_path}: {e}")
            return None, None, None, None


####################CUstom DATASET EMOCIONES RANDOM FOREST###########################
import torchvision.models as models
import torch.nn as nn
# Cargar ResNet18 preentrenada
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

class EmotionDataset_RF(Dataset):
    def __init__(self, data, base_path, transform=None):
        self.data = data.reset_index(drop=True)
        self.base_path = base_path
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Índice {idx} fuera de rango")

        row = self.data.iloc[idx]
        img_path = os.path.join(self.base_path, row["Ruta"]) 

        try:
            if not os.path.exists(img_path):
                print(f"Error: La imagen no existe en {img_path}")
                return torch.zeros(512 + 10), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])  # Asegurar 4 valores

            # Cargar imagen y extraer features con ResNet18
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                img_features = resnet(image).squeeze().flatten()

            # Características adicionales
            required_features = [
                "RMS", "ZCR", "Crest Factor",
                "Standard Deviation of Amplitude", "Spectral Centroid",
                "Spectral Bandwidth", "Spectral Roll-off", "Spectral Flux",
                "VAD", "Spectral Variation",
            ]
            additional_features = torch.tensor(
                row[required_features].values.astype(float), dtype=torch.float32
            )

            # Concatenar las características de imagen y adicionales
            combined_features = torch.cat((img_features, additional_features))

            # Etiquetas de valencia y arousal
            valencia_cols = [f"Valencia_{i/10:.1f}" for i in range(11)]
            arousal_cols = [f"Arousal_{i/10:.1f}" for i in range(11)]
            valencia_value = row[valencia_cols].values.argmax() / 10
            arousal_value = row[arousal_cols].values.argmax() / 10
            valencia_label = torch.tensor([valencia_value], dtype=torch.float32)
            arousal_label = torch.tensor([arousal_value], dtype=torch.float32)

            return image, combined_features, valencia_label, arousal_label

        except Exception as e:
            print(f"Error al cargar la imagen {img_path}: {e}")
            return torch.zeros(512 + 10), torch.zeros(10), torch.tensor([0.0]), torch.tensor([0.0]) 

#################### Custom Dataset SECTIONS ###########################

class CustomDataset_Sections(torch.utils.data.Dataset):
    def __init__(self, data, base_path, transform):
        self.data = data
        self.data = data.reset_index(drop=True)
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Índice {idx} fuera de rango")
        row = self.data.iloc[idx]
        ruta=row["Ruta"]
        img_path = os.path.join(self.base_path, ruta)

        try:
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            required_columns = [
                "RMS",
                "ZCR",
                "Crest Factor",
                "Standard Deviation of Amplitude",
                "Spectral Centroid",
                "Spectral Bandwidth",
                "Spectral Roll-off",
                "Spectral Flux",
                "VAD",
                "Spectral Variation",
            ]
            for col in required_columns:
                if col not in row:
                    raise ValueError(f"Columna {col} no encontrada en el DataFrame.")

            additional_features = row[required_columns].values.astype(float)
            additional_features = torch.tensor(additional_features, dtype=torch.float32)
            
            label_columns = [
            "Break",
            "Pre-Drop",
            "Drop"]
            labels = torch.tensor(row[label_columns].values.astype(int),dtype=torch.long
        )
            return image, additional_features, labels

        except Exception as e:
            raise RuntimeError(
                f"Error procesando el índice {idx}, archivo {img_path}: {e}"
            )