import os
import sys
import pandas as pd
import seaborn as sns
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)


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
base_path = "/content/drive/MyDrive/TFG/data/"
model_path = "/content/drive/MyDrive/TFG/models/best_crnn_genre.pth"
nuevo_csv_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida_test/dataset_genero_test.csv"
mean = [0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std = [0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
hidden_size = 256
additional_features_dim = 12
num_classes = 6

def main():
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
        nuevo_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
    )

    # Predicciones
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in nuevo_loader:
            images, features, labels, img_paths = batch  # Desempaquetar los cuatro elementos
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

    outputs = model(images, features)
    preds = torch.argmax(outputs, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

    # Si necesitas usar las rutas de las imágenes, puedes hacer algo como:
    for img_path in img_paths:
        print(f"Predicción para la imagen: {img_path}")
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')
    plt.show()

    # Reporte de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(all_labels, all_preds))

    # Guardar predicciones en un archivo CSV
    data["Predicciones"] = all_preds
    output_csv_path = "/content/drive/MyDrive/TFG/predicciones_con_matriz_confusion.csv"
    data.to_csv(output_csv_path, index=False)
    print(f"Predicciones guardadas en {output_csv_path}")

if __name__ == "__main__":
    main()