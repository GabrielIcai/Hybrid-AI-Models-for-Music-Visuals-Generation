import torch
import pandas as pd
import numpy as np
import os
import seaborn as sns
import sys
import torch
from collections import Counter, defaultdict
import re
from PIL import Image

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.preprocessing import CustomDataset_s, normalize_columns, load_data, c_transform
from src.training import collate_fn_s, extract_song_name
from src.models.genre_model import CNN_LSTM_genre

# Configuración inicial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Parámetros
base_path = "/content/drive/MyDrive/TFG/data/"
model_path = "/content/drive/MyDrive/TFG/models/best_cnn_lstm_genre.pth"
csv_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida_test/dataset_genero_test.csv"
mean = [0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std = [0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
columns_to_normalize = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
class_names = ["Afro House", "Ambient", "Deep House", "Techno", "Trance", "Progressive House"]

# Cargar y preparar datos
data = load_data(csv_path)
pd.set_option("display.max_columns", None)
print(data.head())

# Verificar columna Song ID
assert "Song ID" in data.columns, "El CSV tiene que contener la columna 'Song ID' para agrupar por canciones"

# Preprocesamiento
data["Ruta"] = data["Ruta"].str.replace("\\", "/")
data["Ruta"] = base_path + data["Ruta"]
normalize_columns(data, columns_to_normalize)

# Cargar modelo
model = CNN_LSTM_genre(num_classes=len(class_names), additional_features_dim=12, hidden_size=256)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Dataset y DataLoader
test_transform = c_transform(mean, std)
test_dataset = CustomDataset_s(data, base_path, transform=test_transform)
test_loader = DataLoader(
    test_dataset, 
    batch_size=128, 
    collate_fn=collate_fn_s,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Después de completar la inferencia
all_preds = []
all_labels = []
all_probabilities = []
song_group_predictions = defaultdict(list)  # Almacena las predicciones por canción

with torch.no_grad():
    for images, additional_features, labels, image_paths in test_loader:
        images = images.to(device)
        additional_features = additional_features.to(device)
        labels = labels.to(device)

        outputs = model(images, additional_features)
        preds = torch.argmax(outputs, dim=1)
        labels_grouped = torch.argmax(labels, dim=1)
        probabilities = torch.softmax(outputs, dim=1)

        # Agrupar las predicciones por canción
        for i, image_path in enumerate(image_paths):
            song_name = extract_song_name(image_path)
            if song_name:
                song_group_predictions[song_name].append(preds[i].item())

        all_probabilities.extend(probabilities.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_grouped.cpu().numpy())

# Calcular la predicción más común (moda) por canción
final_song_predictions = {}
for song_name, preds_list in song_group_predictions.items():
    # Calculamos la moda de las predicciones para cada canción
    most_common_pred = stats.mode(preds_list)[0][0]
    final_song_predictions[song_name] = most_common_pred

# Mostrar las predicciones finales por canción
print("\nPredicciones finales por canción:")
for song_name, pred in final_song_predictions.items():
    print(f"Canción: {song_name}, Predicción más común: {class_names[pred]}")