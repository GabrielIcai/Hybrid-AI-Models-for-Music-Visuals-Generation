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
from src.preprocessing import CustomDataset, normalize_columns, load_data, c_transform
from src.training import collate_fn
from src.models.genre_model import CRNN, CNN_LSTM_genre

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Rutas
base_path = "/content/drive/MyDrive/TFG/data/"
model_path = "/content/drive/MyDrive/TFG/models/best_cnn_lstm_genre.pth"
csv_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida_test/dataset_genero_test.csv"
output_csv = "/content/drive/MyDrive/TFG/predicciones_canciones_LSTM.csv"

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
data=data.head(100)
normalize_columns(data, columns)

# Agrupar por "song_ID"
canciones_unicas = data["song_ID"].unique()

# Transformación
test_transform = c_transform(mean, std)

# Archivo CSV donde guardaremos los resultados
csv_headers = ["song_ID", "fragmento", "etiqueta_real", "prediccion", "probabilidades"]
with open(output_csv, "w") as f:
    f.write(",".join(csv_headers) + "\n")

# Inferencia por canción
with torch.no_grad():
    for song_id in canciones_unicas:
        print(f"\nProcesando canción ID: {song_id}")

        # Filtrar fragmentos de la canción actual
        data_cancion = data[data["song_ID"] == song_id].reset_index(drop=True)
        test_dataset = CustomDataset(data_cancion, base_path, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

        # Predicción por fragmento
        for idx, (image, additional_features, label) in enumerate(test_loader):
            image = image.to(device)
            additional_features = additional_features.to(device)
            label = label.to(device)

            output = model(image, additional_features)
            pred = torch.argmax(output, dim=1).cpu().item()
            label_real = torch.argmax(label, dim=1).cpu().item()
            probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()

            # Guardar resultados en CSV
            fragmento = f"fragmento_{idx}"
            prob_str = ";".join([str(p) for p in probabilities])
            with open(output_csv, "a") as f:
                f.write(f"{song_id},{fragmento},{label_real},{pred},{prob_str}\n")

print("\n Predicciones guardadas en:", output_csv)