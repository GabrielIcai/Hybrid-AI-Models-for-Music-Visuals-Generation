import torch
import pandas as pd
import numpy as np
import os
import sys
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from src.preprocessing import CustomDataset, normalize_columns, load_data, c_transform
from src.training import collate_fn
from src.models.genre_model import CRNN

# Define las transformaciones
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

def c_transform(mean, std):
    return Compose([ToTensor(), Normalize(mean=mean, std=std)])

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    data["Ruta"] = data["Ruta"].str.replace("\\", "/")
    data["Ruta"] = base_path + data["Ruta"]
    normalize_columns(data, columns)
    return data

def load_model(model_path, num_classes, additional_features_dim, hidden_size):
    model = CRNN(num_classes, additional_features_dim, hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="viridis")
    plt.title("Matriz de Confusión")
    plt.show()

def predict(model, dataloader, device):
    model.eval()  # Establecer el modelo en modo evaluación
    y_true = []
    y_pred = []
    
    with torch.no_grad():  # Desactivar el cálculo de gradientes para predicciones
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Hacer predicciones
            outputs = model(inputs)
            
            # Convertir las salidas a etiquetas de clase (por ejemplo, usando argmax)
            _, predicted = torch.max(outputs, 1)
            
            # Almacenar las etiquetas verdaderas y predichas
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar los datos
    data = load_data(nuevo_csv_path)
    transform = c_transform(mean, std)
    dataset = CustomDataset(data, base_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=2)
    print(data.head(100))

    # Cargar el modelo
    model = load_model(model_path, num_classes, additional_features_dim, hidden_size).to(device)

    # Realizar predicciones
    y_true, y_pred = predict(model, dataloader, device)

    # Definir los nombres de las clases
    class_names = ["Clase 0", "Clase 1", "Clase 2", "Clase 3", "Clase 4", "Clase 5"]

    # Imprimir el reporte de clasificación
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Mostrar la matriz de confusión
    plot_confusion_matrix(y_true, y_pred, class_names)

if __name__ == "__main__":
    main()
