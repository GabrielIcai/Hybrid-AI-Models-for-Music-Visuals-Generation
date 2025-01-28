import torch
import pandas as pd
import numpy as np
import os
import seaborn as sns
import sys
import torch

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.preprocessing import CustomDataset_s, normalize_columns, load_data, c_transform
from src.training import collate_fn
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
assert "Song ID" in data.columns, "El CSV debe contener la columna 'Song ID' para agrupar por canciones"

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
    collate_fn=collate_fn,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

from collections import Counter

# Después de completar la inferencia
all_preds = []
all_labels = []
all_probabilities = []
song_ids = data["Song ID"].tolist()  # Asegúrate de que el dataset tenga una columna 'Song ID'

song_group_predictions = {}  # Diccionario para almacenar predicciones por canción
song_group_labels = {}       # Diccionario para etiquetas reales por canción

with torch.no_grad():
    for idx, (images, additional_features, labels) in enumerate(test_loader):
        images = images.to(device)
        additional_features = additional_features.to(device)
        labels = labels.to(device)

        outputs = model(images, additional_features)
        preds = torch.argmax(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1)

        # Obtener IDs de las canciones para este batch
        batch_song_ids = song_ids[idx * test_loader.batch_size : (idx + 1) * test_loader.batch_size]
        
        # Agrupar predicciones y etiquetas por canción
        for i, song_id in enumerate(batch_song_ids):
            if song_id not in song_group_predictions:
                song_group_predictions[song_id] = []
                song_group_labels[song_id] = []

            song_group_predictions[song_id].append(preds[i].item())
            song_group_labels[song_id].append(torch.argmax(labels[i]).item())

# Obtener la predicción final por canción (la más frecuente)
final_song_predictions = {}
final_song_labels = {}

for song_id, predictions in song_group_predictions.items():
    # Predicción más frecuente
    most_common_prediction = Counter(predictions).most_common(1)[0][0]
    final_song_predictions[song_id] = most_common_prediction
    
    # Etiqueta real (asumimos que es la misma para todos los fragmentos de la canción)
    final_song_labels[song_id] = song_group_labels[song_id][0]

# Convertir a listas para métricas
final_preds = list(final_song_predictions.values())
final_labels = list(final_song_labels.values())

# Generar matriz de confusión
conf_matrix = confusion_matrix(final_labels, final_preds)
print("\nMatriz de confusión (por canción):")
print(conf_matrix)

# Reporte de clasificación
print("\nReporte de clasificación (por canción):")
print(classification_report(final_labels, final_preds, target_names=class_names))

# Visualizar matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas Reales")
plt.title("Matriz de Confusión (por canción)")

image_path = "/content/drive/MyDrive/TFG/matriz_confusion_generos_lstm_por_cancion.png"
plt.savefig(image_path)
plt.close()
