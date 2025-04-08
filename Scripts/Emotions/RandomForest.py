import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)
from src.preprocessing import (
    load_data,
    split_dataset,
    c_transform,
)
from src.preprocessing.data_loader import load_data, split_dataset
from src.training import collate_fn_emotions
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.preprocessing.custom_dataset import EmotionDataset_RF
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import torch
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader

def extract_features(dataloader):
    X, y_v, y_a = [], [], []
    
    for images, features, valence, arousal in dataloader:
        X.append(features.numpy())
        y_v.append(valence.numpy())
        y_a.append(arousal.numpy())

    X = np.concatenate(X)
    y_v = np.concatenate(y_v)
    y_a = np.concatenate(y_a)
    
    return X, y_v, y_a


# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
    learning_rate = 0.001
    weight_decay = 1e-5
    data_path = "/content/drive/MyDrive/TFG/images/espectrogramas_normalizados_emociones_estructura/dataset_emociones_secciones.csv"
    base_path = "/content/drive/MyDrive/TFG/images/"
    mean=[0.676956295967102, 0.2529653012752533, 0.4388839304447174]
    std=[0.21755781769752502, 0.15407244861125946, 0.07557372003793716]

    data = load_data(data_path)
    data["Ruta"] = data["Ruta"].str.replace("\\", "/")
    data["Ruta"] = base_path + data["Ruta"]
    data["Ruta"] = data["Ruta"].str.replace("espectrogramas_salida_secciones_2", "espectrogramas_normalizados_emociones_estructura")
    train_data, test_data = split_dataset(data)
    
    # Transformaciones
    train_transform = c_transform(mean, std)
    test_transform = c_transform(mean, std)
    
    # Crear datasets
    train_dataset = EmotionDataset_RF(train_data, base_path, transform=train_transform)
    test_dataset = EmotionDataset_RF(test_data, base_path, transform=test_transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_emotions)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_emotions) 

    X_train, y_train_v, y_train_a = extract_features(train_loader)
    X_test, y_test_v, y_test_a = extract_features(test_loader)

    # Definir modelos de Random Forest
    rf_valencia = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_arousal = RandomForestRegressor(n_estimators=100, random_state=42)

    # Entrenar los modelos
    rf_valencia.fit(X_train, y_train_v)
    rf_arousal.fit(X_train, y_train_a)

    # Predicciones
    y_pred_v = rf_valencia.predict(X_test)
    y_pred_a = rf_arousal.predict(X_test)

    # Evaluación con Mean Squared Error (MSE)
    mse_v = mean_squared_error(y_test_v, y_pred_v)
    mse_a = mean_squared_error(y_test_a, y_pred_a)

    print(f"MSE Valencia: {mse_v:.4f}")
    print(f"MSE Arousal: {mse_a:.4f}")

    # Gráfico de dispersión para Valencia
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_v, y_pred_v, color='blue', alpha=0.5)
    plt.plot([y_test_v.min(), y_test_v.max()], [y_test_v.min(), y_test_v.max()], 'k--', lw=2)  # Línea diagonal
    plt.xlabel('Valores reales de Valencia')
    plt.ylabel('Predicciones de Valencia')
    plt.title('Predicciones vs Valores Reales (Valencia)')

    # Guardar la imagen
    plt.savefig('/content/drive/MyDrive/dispersion_valencia_RF.png')
    plt.close()

    # Gráfico de dispersión para Arousal
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_a, y_pred_a, color='green', alpha=0.5)
    plt.plot([y_test_a.min(), y_test_a.max()], [y_test_a.min(), y_test_a.max()], 'k--', lw=2)  # Línea diagonal
    plt.xlabel('Valores reales de Arousal')
    plt.ylabel('Predicciones de Arousal')
    plt.title('Predicciones vs Valores Reales (Arousal)')

    # Guardar la imagen
    plt.savefig('/content/drive/MyDrive/dispersion_arousal.png')
    plt.close()
