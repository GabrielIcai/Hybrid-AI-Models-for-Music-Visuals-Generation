import torch
import pandas as pd
import numpy as np
import os
import seaborn as sns
import sys
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.preprocessing import CustomDataset_s, normalize_columns, load_data, c_transform,CustomDataset
from src.training import collate_fn_s,collate_fn
from src.models.genre_model import CRNN, CNN_LSTM_genre

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Rutas
base_path = "/content/drive/MyDrive/TFG/data/"
model_path = "/content/drive/MyDrive/TFG/models/best_cnn_lstm_genre.pth"
csv_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida_test/dataset_test.csv"
output_csv_path = "/content/drive/MyDrive/TFG/predicciones_canciones_LSTM.csv"

# Normalización
mean = [0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std = [0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
num_classes = 6
class_names = ["Afro House", "Ambient", "Deep House", "Techno", "Trance", "Progressive House"]

model = CNN_LSTM_genre(num_classes=num_classes, additional_features_dim=12, hidden_size=256)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Cargar dataset
data = load_data(csv_path)
data["Ruta"] = data["Ruta"].str.replace("\\", "/")
data["Ruta"] = base_path + data["Ruta"]
data=data.head(20)
normalize_columns(data, columns)
print(data.head(20))

# Agrupar por "song_ID"
canciones_unicas = data["Song ID"].unique()

# Transformación
test_transform = c_transform(mean, std)
test_dataset = CustomDataset(data, base_path, transform=test_transform)
test_loader = DataLoader( test_dataset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=2, pin_memory=True)

# Crear listas vacías para almacenar las predicciones, etiquetas reales y probabilidades
all_preds = []
all_labels = []
all_probabilities = []

# Procesar las imágenes y almacenar los resultados
with torch.no_grad():
    for images, additional_features, labels in test_loader:
        images = images.to(device)
        additional_features = additional_features.to(device)
        labels = labels.to(device)

        outputs = model(images, additional_features)
        preds = torch.argmax(outputs, dim=1)
        labels_grouped = torch.argmax(labels, dim=1)
        probabilities = torch.softmax(outputs, dim=1)

        all_probabilities.extend(probabilities.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_grouped.cpu().numpy())

# Ahora creamos un DataFrame con las predicciones, etiquetas reales y probabilidades
results = {
    'Predicción': all_preds,
    'Etiqueta Real': all_labels,
    'Probabilidades': [prob.tolist() for prob in all_probabilities]  # Convertir las probabilidades en listas
}

df_results = pd.DataFrame(results)

# Guardar el DataFrame en un archivo CSV
df_results.to_csv(output_csv_path, index=False)

print(f"Predicciones guardadas en: {output_csv_path}")
