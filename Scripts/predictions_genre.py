import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_path not in sys.path:
  sys.path.append(repo_path)

from src.training import collate_fn
from src.models.genre_model import CRNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from src.preprocessing import (
  CustomDataset,
  normalize_columns,
  c_transform,
)
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
from src.preprocessing.custom_dataset import CustomDataset


# Parámetros
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
base_path = "/content/drive/MyDrive/TFG/data/"  # Use sus rutas
model_path = "/content/drive/MyDrive/TFG/models/best_crnn_genre.pth"  # Use sus rutas
nuevo_csv_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida_test/dataset_genero_test.csv"  # Use sus rutas
mean = [0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std = [0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
hidden_size = 256
additional_features_dim = 12
num_classes = 6

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def main():
  # Cargar el modelo
  model = CRNN(num_classes, additional_features_dim, hidden_size)
  model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
  model.to(device)
  model.eval()
  print("Modelo cargado.")

  # Preprocesar nuevos datos
  data = pd.read_csv(nuevo_csv_path)
  data["Ruta"] = data["Ruta"].str.replace("\\", "/")
  data["Ruta"] = base_path + data["Ruta"]
  normalize_columns(data, columns)

  # Crear dataset y DataLoader
  test_transform = c_transform(mean, std)
  nuevo_dataset = CustomDataset(data, base_path, transform=test_transform)
  nuevo_loader = DataLoader(
      nuevo_dataset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=2, pin_memory=True
  )

  # Predicciones
  all_preds = []
  all_labels = []
  fragment_preds = []  # Aquí almacenaremos las predicciones por fragmento de tres
  fragment_labels = []  # Aquí almacenaremos las etiquetas por fragmento de tres

  with torch.no_grad():
    for batch in nuevo_loader:
      images, features, labels = batch  # Desempaquetar los cuatro elementos
      images = images.to(device)
      features = features.to(device)
      labels = labels.to(device)

      # Predicción en fragmentos de tres en tres
      outputs = model(images, features)
      preds = torch.argmax(outputs, dim=1)

      # Almacenamos las predicciones y etiquetas por fragmento
      fragment_preds.append(preds.cpu().numpy())
      fragment_labels.append(labels.cpu().numpy())

  # Convertir las etiquetas one-hot a enteros (índice de la clase)

