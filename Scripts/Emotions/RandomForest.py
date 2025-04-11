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
from sklearn.metrics import r2_score
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
from sklearn.metrics import mean_absolute_error
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
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    
    def check_file_exists(file_path):
        return os.path.isfile(file_path)

    # Cargar los datos y limpiar las rutas no encontradas
    def load_and_clean_data(data_path, base_path):
        data = pd.read_csv(data_path)
        data["Ruta"] = data["Ruta"].str.replace("\\", "/")
        data["Ruta"] = base_path + data["Ruta"]
        data["Ruta"] = data["Ruta"].str.replace("espectrogramas_salida_secciones_2", "espectrogramas_normalizados_emociones_estructura")
        
        # Filtrar las rutas que no existen
        data = data[data["Ruta"].apply(check_file_exists)]
        
        print(f"Total de archivos válidos: {len(data)}")
        return data

    # Llamar a la función para cargar y limpiar los datos
    data = load_and_clean_data(data_path, base_path)
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
    mae_v = mean_absolute_error(y_test_v, y_pred_v)
    mae_a = mean_absolute_error(y_test_a, y_pred_a)
    r2_v = r2_score(y_test_v, y_pred_v)
    r2_a = r2_score(y_test_a, y_pred_a)

    print(f"MSE Valencia: {mse_v:.4f}")
    print(f"MSE Arousal: {mse_a:.4f}")

    # Gráfico de dispersión para Valencia
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_v, y_pred_v, color='blue', alpha=0.5)
    plt.plot([y_test_v.min(), y_test_v.max()], [y_test_v.min(), y_test_v.max()], 'k--', lw=2)  # Línea diagonal
    plt.xlabel('Valores reales de Valencia')
    plt.ylabel('Predicciones de Valencia')
    plt.title('Predicciones vs Valores Reales (Valencia)')
    plt.savefig('/content/drive/MyDrive/dispersion_valencia_RF.png')
    plt.close()

    # Gráfico de dispersión para Arousal
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_a, y_pred_a, color='green', alpha=0.5)
    plt.plot([y_test_a.min(), y_test_a.max()], [y_test_a.min(), y_test_a.max()], 'k--', lw=2)  # Línea diagonal
    plt.xlabel('Valores reales de Arousal')
    plt.ylabel('Predicciones de Arousal')
    plt.title('Predicciones vs Valores Reales (Arousal)')
    plt.savefig('/content/drive/MyDrive/dispersion_arousal.png')
    plt.close()
    errors_v = y_test_v - y_pred_v
    errors_a = y_test_a - y_pred_a

    plt.figure(figsize=(8, 6))
    plt.hist(errors_v, bins=30, alpha=0.7, label='Valencia', color='blue')
    plt.hist(errors_a, bins=30, alpha=0.7, label='Arousal', color='green')
    plt.xlabel('Errores')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.title('Distribución de Errores')
    plt.savefig('/content/drive/MyDrive/distribucion_errores.png')
    plt.close()

    metrics = {
        'Métrica': ['MSE_Valencia', 'MSE_Arousal', 'MAE_Valencia', 'MAE_Arousal', 'R2_Valencia', 'R2_Arousal'],
        'Valor': [mse_v, mse_a, mae_v, mae_a, r2_v, r2_a]
    }

    # Guardar los modelos entrenados en Google Drive
    joblib.dump(rf_valencia, "/content/drive/MyDrive/modelo_valencia.pkl")
    joblib.dump(rf_arousal, "/content/drive/MyDrive/modelo_arousal.pkl")

    # Convertir a DataFrame y guardar en CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('/content/drive/MyDrive/metrics_RF.csv', index=False)

