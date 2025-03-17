import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
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
from src.preprocessing.custom_dataset import EmotionDataset


mean=[0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std=[0.21755781769752502, 0.15407244861125946, 0.07557372003793716]

# Cargar datos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
learning_rate = 0.001
weight_decay = 1e-5
data_path = "/content/drive/MyDrive/TFG/images/espectrogramas_normalizados_emociones_estructura/dataset_emociones_secciones.csv"
base_path = "/content/drive/MyDrive/TFG/images/"
epochs = 50
patience = 5
best_val_loss = float("inf")
early_stop_counter = 0
epochs_list = []
train_losses, val_losses = [], []
val_maes_va,val_maes_ar = [], []
rmse_arousal, rmse_valence = [], []
r2_valence, r2_arousal = [], []
data = load_data(data_path)
data["Ruta"] = data["Ruta"].str.replace("\\", "/")
data["Ruta"] = base_path + data["Ruta"]
data["Ruta"] = data["Ruta"].str.replace("espectrogramas_salida_secciones_2", "espectrogramas_normalizados_emociones_estructura")
train_data, test_data = split_dataset(data)

# Transformaciones
train_transform = c_transform(mean, std)
test_transform = c_transform(mean, std)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)

# Función para extraer características
def extract_features(data_loader, model, device):
    model.eval()
    features, labels_ar, labels_va = [], [], []
    with torch.no_grad():
        for images, arousal, valence in data_loader:
            images = images.to(device)
            feats = model(images).cpu().numpy()
            features.extend(feats)
            labels_ar.extend(arousal.numpy())
            labels_va.extend(valence.numpy())
    return np.array(features), np.array(labels_ar), np.array(labels_va)

# Cargar datasets
train_dataset = EmotionDataset(train_data, base_path, transform=train_transform)
test_dataset = EmotionDataset(test_data, base_path, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_emotions)
val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_emotions)

# Extraer características
extractor = FeatureExtractor().to(device)
train_features, train_labels_ar, train_labels_va = extract_features(train_loader, extractor, device)
val_features, val_labels_ar, val_labels_va = extract_features(val_loader, extractor, device)

# Normalizo
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

# Guardar el scaler para futuras predicciones
joblib.dump(scaler, "scaler.pkl")

# Entrenar modelos Random Forest
rf_arousal = RandomForestRegressor(n_estimators=100, random_state=42)
rf_valence = RandomForestRegressor(n_estimators=100, random_state=42)
rf_arousal.fit(train_features, train_labels_ar)
rf_valence.fit(train_features, train_labels_va)

# Evaluación
val_preds_ar = rf_arousal.predict(val_features)
val_preds_va = rf_valence.predict(val_features)

mae_ar = mean_absolute_error(val_labels_ar, val_preds_ar)
mae_va = mean_absolute_error(val_labels_va, val_preds_va)
r2_ar = r2_score(val_labels_ar, val_preds_ar)
r2_va = r2_score(val_labels_va, val_preds_va)

print(f"MAE Arousal: {mae_ar:.4f}, R2 Arousal: {r2_ar:.4f}")
print(f"MAE Valence: {mae_va:.4f}, R2 Valence: {r2_va:.4f}")

# Guardar modelos
joblib.dump(rf_arousal, "random_forest_arousal.pkl")
joblib.dump(rf_valence, "random_forest_valence.pkl")

# Guardar gráficas de predicción
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Ajustar límites en función de los valores reales y predichos
min_ar, max_ar = min(val_labels_ar.min(), val_preds_ar.min()), max(val_labels_ar.max(), val_preds_ar.max())
min_va, max_va = min(val_labels_va.min(), val_preds_va.min()), max(val_labels_va.max(), val_preds_va.max())

axs[0].scatter(val_labels_ar, val_preds_ar, alpha=0.5)
axs[0].set_xlabel("Real Arousal")
axs[0].set_ylabel("Predicted Arousal")
axs[0].set_title("Predicción Arousal")
axs[0].plot([min_ar, max_ar], [min_ar, max_ar], '--r')

axs[1].scatter(val_labels_va, val_preds_va, alpha=0.5)
axs[1].set_xlabel("Real Valence")
axs[1].set_ylabel("Predicted Valence")
axs[1].set_title("Predicción Valence")
axs[1].plot([min_va, max_va], [min_va, max_va], '--r')

# Guardar figuras en archivos
plt.savefig("predicciones_arousal_valence.png")