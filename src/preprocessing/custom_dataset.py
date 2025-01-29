import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


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

            required_columns = ["RMS","ZCR","Mean Absolute Amplitude","Crest Factor","Standard Deviation of Amplitude",
                "Spectral Centroid","Spectral Bandwidth","Spectral Roll-off","Spectral Flux","VAD","Spectral Variation",
                "Tempo"]
            for col in required_columns:
                if col not in row:
                    raise ValueError(f"Columna {col} no encontrada en el DataFrame.")

            additional_features = row[required_columns].values.astype(float)
            additional_features = torch.tensor(additional_features, dtype=torch.float32)
            #Selecciono 
            label_columns = ["Afro House","Ambient","Deep House","Techno","Trance","Progressive House"]
            labels = torch.tensor(row[label_columns].values.astype(int),dtype=torch.long
        )
            song_id = self.data.iloc[idx]["Song ID"] if "Song ID" in self.data.columns else None

            if song_id is not None:
                return image, additional_features, labels, song_id
            else:
                return image, additional_features, labels

        except Exception as e:
            raise RuntimeError(
                f"Error procesando el índice {idx}, archivo {img_path}: {e}"
            )




#PRUEBAS CANCIONES
class CustomDataset_s(torch.utils.data.Dataset):
    def __init__(self, data, base_path, transform):
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
            print(f"Cargando imagen desde: {img_path}")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            required_columns = [
                "RMS", "ZCR", "Mean Absolute Amplitude", "Crest Factor",
                "Standard Deviation of Amplitude", "Spectral Centroid",
                "Spectral Bandwidth", "Spectral Roll-off", "Spectral Flux",
                "VAD", "Spectral Variation", "Tempo",
            ]
            missing_columns = [col for col in required_columns if col not in row]
            if missing_columns:
                raise ValueError(f"Faltan columnas: {', '.join(missing_columns)}")
            additional_features = torch.tensor(
                row[required_columns].values.astype(float), dtype=torch.float32
            )
            label_columns = [
                "Afro House", "Ambient", "Deep House",
                "Techno", "Trance", "Progressive House",
            ]
            labels = torch.tensor(
                row[label_columns].values.astype(int), dtype=torch.long
            )

            #SONG ID ES OPCIONAL
            song_id = row["Song ID"] if "Song ID" in row else None

            if song_id is not None:
                print(song_id)
                return image, additional_features, labels, song_id
            else:
                return image, additional_features, labels

        except Exception as e:
            raise RuntimeError(
                f"Error procesando el índice {idx}, archivo {img_path}: {e}"
            )
