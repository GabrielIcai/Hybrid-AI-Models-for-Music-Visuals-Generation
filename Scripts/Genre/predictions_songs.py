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
from src.preprocessing import CustomDataset, normalize_columns, load_data, c_transform
from src.training import collate_fn
from src.models.genre_model import CNN_LSTM_genre

# Configuración inicial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Parámetros
base_path = "/content/drive/MyDrive/TFG/data/"
model_path = "/content/drive/MyDrive/TFG/models/best_cnn_lstm_genre.pth"
csv_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida_test/dataset_genero_test - dataset_genero_test(1).csv"
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
test_dataset = CustomDataset(data, base_path, transform=test_transform)
test_loader = DataLoader(
    test_dataset, 
    batch_size=128, 
    collate_fn=collate_fn,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Realizar inferencia
all_preds = []
all_labels = []
all_song_ids = []

with torch.no_grad():
    for images, additional_features, labels in test_loader:
        # Mover datos al dispositivo
        images = images.to(device)
        additional_features = additional_features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images, additional_features)
        
        # Obtener predicciones
        preds = torch.argmax(outputs, dim=1)
        labels_indices = torch.argmax(labels, dim=1)
        
        # Guardar resultados
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_indices.cpu().numpy())
        all_song_ids.extend(test_dataset.data["Song ID"].iloc[test_dataset.indices].tolist())

# Agrupar predicciones por canción
song_predictions = {}
song_true_labels = {}

for song_id, pred, true_label in zip(all_song_ids, all_preds, all_labels):
    if song_id not in song_predictions:
        song_predictions[song_id] = []
        song_true_labels[song_id] = true_label  # Asumimos que todos los fragmentos tienen la misma etiqueta
    
    song_predictions[song_id].append(pred)

# Votación mayoritaria por canción
final_predictions = []
final_true_labels = []

for song_id in song_predictions:
    # Obtener la moda de las predicciones
    majority_vote = np.argmax(np.bincount(song_predictions[song_id]))
    final_predictions.append(majority_vote)
    final_true_labels.append(song_true_labels[song_id])

# Evaluación a nivel de canción
print("\nEvaluación a nivel de canción completa:")
print("--------------------------------------")

# Directorio para guardar resultados
output_dir = "/content/drive/MyDrive/TFG/resultados/"
os.makedirs(output_dir, exist_ok=True)  # Crear directorio si no existe

# Matriz de confusión
conf_matrix = confusion_matrix(final_true_labels, final_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas Reales")
plt.title("Matriz de Confusión LSTM - Canción")
plt.savefig(os.path.join(output_dir, "matriz_confusion_canciones_LSTM.png"))
plt.close()

# Reporte de clasificación
class_report = classification_report(final_true_labels, final_predictions, target_names=class_names, output_dict=True)
with open(os.path.join(output_dir, "reporte_clasificacion.txt"), "w") as f:
    f.write(classification_report(final_true_labels, final_predictions, target_names=class_names))

# Gráficas adicionales

# 1. Distribución de Predicciones por Canción
plt.figure(figsize=(12, 6))
for i, song_id in enumerate(song_predictions):
    plt.bar(i, len(song_predictions[song_id]), label=f'{song_id} - {class_names[final_predictions[i]]}')
plt.xlabel("Canciones")
plt.ylabel("Número de Fragmentos")
plt.title("Distribución de Predicciones por Canción LSTM")
plt.xticks([])
plt.legend()
plt.savefig(os.path.join(output_dir, "distribucion_predicciones_por_cancion_LSTM.png"))
plt.close()

# 2. Precisión por Clase
precision_values = [class_report[class_name]['precision'] for class_name in class_names]
plt.figure(figsize=(10, 6))
sns.barplot(x=class_names, y=precision_values, palette="viridis")
plt.xlabel("Clases")
plt.ylabel("Precisión")
plt.title("Precisión por Clase LSTM")
plt.savefig(os.path.join(output_dir, "precision_por_clase_LSTM.png"))
plt.close()

# 3. Distribución de Etiquetas Reales
plt.figure(figsize=(10, 6))
sns.countplot(x=final_true_labels, palette="Set2")
plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
plt.xlabel("Clases")
plt.ylabel("Número de Canciones")
plt.title("Distribución de Etiquetas Reales LSTM")
plt.savefig(os.path.join(output_dir, "distribucion_etiquetas_reales_LSTM.png"))
plt.close()

print(f"Resultados guardados en: {output_dir}")