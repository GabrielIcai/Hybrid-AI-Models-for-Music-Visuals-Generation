import torch
import pandas as pd
import numpy as np
import os
import seaborn as sns
import sys
import torch
from collections import Counter, defaultdict
import re
from scipy import stats

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
csv_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida_test/dataset_test.csv"
mean = [0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std = [0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
columns_to_normalize = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
class_names = ["Afro House", "Ambient", "Deep House", "Techno", "Trance", "Progressive House"]


data = load_data(csv_path)
pd.set_option("display.max_columns", None)
print(data.head())

# Verifico Song ID
assert "Song ID" in data.columns, "El CSV tiene 'Song ID'"

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

all_preds = {}
all_labels = {}
all_probabilities = {}

with torch.no_grad():
    for images, additional_features, labels, song_ids in test_loader:
        batch_size = len(song_ids)  # Número de canciones en este batch
        
        for i in range(batch_size):
            song_images = images[i].to(device)  # Todos los fragmentos de una canción
            song_add_feats = additional_features[i].to(device)
            song_label = labels[i].to(device)
            song_id = song_ids[i]

            outputs = model(song_images, song_add_feats)  # Predicción para todos los fragmentos
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)  # Predicción para cada fragmento
            
            # Guardar predicciones
            if song_id not in all_preds:
                all_preds[song_id] = []
                all_probabilities[song_id] = []
                all_labels[song_id] = song_label.cpu().numpy()  # Etiqueta real

            all_preds[song_id].extend(preds.cpu().numpy())
            all_probabilities[song_id].extend(probabilities.cpu().numpy())

# Consolidar predicciones por canción
final_preds = []
final_labels = []

for song_id in all_preds:
    most_common_pred = Counter(all_preds[song_id]).most_common(1)[0][0]
    
    avg_probabilities = np.mean(all_probabilities[song_id], axis=0)
    best_pred = np.argmax(avg_probabilities)

    final_pred = best_pred  # Se elige el promedio de probabilidades
    # final_pred = most_common_pred  # Si prefieres la moda

    final_preds.append(final_pred)
    final_labels.append(all_labels[song_id])

# Matriz de confusión y métricas
conf_matrix = confusion_matrix(final_labels, final_preds)
print("\nMatriz de confusión:")
print(conf_matrix)

print("\nReporte de clasificación:")
print(classification_report(final_labels, final_preds))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas Reales")
plt.title("Matriz de Confusión por Canción")

image_path = "/content/drive/MyDrive/TFG/matriz_confusion_canciones.png"
plt.savefig(image_path)