import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)


from src.models.emotions_model import ResNetCRNNEmotionModel
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
import torch.optim as optim
from torch.utils.data import DataLoader
from src.preprocessing.custom_dataset import EmotionDataset
from src.training.trainer_emotions import trainer_emotions, validate_emotions
from src.utils import plot_scatter, plot_and_save_residuals

mean=[0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std=[0.21755781769752502, 0.15407244861125946, 0.07557372003793716]

# Cargar datos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
learning_rate = 0.001
weight_decay = 1e-5
data_path = "/content/drive/MyDrive/TFG/images/espectrogramas_normalizados_emociones_estructura/dataset_emociones_secciones.csv"
base_path = "/content/drive/MyDrive/TFG/images/"


def main():
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

    # Crear datasets
    train_dataset = EmotionDataset(train_data, base_path, transform=train_transform)
    test_dataset = EmotionDataset(test_data, base_path, transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_emotions)
    val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_emotions)

    # Modelo y optimización
    model = ResNetCRNNEmotionModel().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Total datos: {len(data)}")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Train loader batches: {len(train_loader)}")

    for epoch in range(epochs):

        all_preds_ar = []
        all_preds_va = []
        all_labels_ar = []
        all_labels_va = []


        # ENTRENAMIENTO
        train_loss, train_rmse_ar, train_rmse_va = trainer_emotions(
            model, train_loader, optimizer, criterion, device
        )

        # VALIDACIÓN
        val_loss, val_rmse_ar, val_rmse_va, val_preds_ar, val_preds_va, val_labels_ar, val_labels_va = validate_emotions(
            model, val_loader, criterion, device
        )

        all_preds_ar.extend(val_preds_ar)
        all_preds_va.extend(val_preds_va)

        # ETIQUETAS A ÍNDICES
        val_labels_ar = np.array(val_labels_ar)
        val_labels_va = np.array(val_labels_va)
        
        all_labels_ar.extend(val_labels_ar) 
        all_labels_va.extend(val_labels_va)
        
        # Métricas de la epoca
        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)


        # métricas adicionales
        val_mae_ar = mean_absolute_error(val_labels_ar, val_preds_ar)
        val_mae_va = mean_absolute_error(val_labels_va, val_preds_va)
        val_r2_ar = r2_score(val_labels_ar, val_preds_ar)
        val_r2_va = r2_score(val_labels_va, val_preds_va)

        # Guardo métricas en listas
        rmse_arousal.append(val_rmse_ar)
        r2_arousal.append(val_r2_ar)
        r2_valence.append(val_r2_va)
        rmse_valence.append(val_rmse_va)
        val_maes_ar.append(val_mae_ar)
        val_maes_va.append(val_mae_va)

        print(f"Epoch {epoch + 1}: val_preds_ar {len(val_preds_ar)}, val_preds_va {len(val_preds_va)}")
        print(f"Epoch {epoch + 1}: val_labels_ar {len(val_labels_ar)}, val_labels_va {len(val_labels_va)}")

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | Train RMSE AR: {train_rmse_ar:.4f}% | Train RMSE VA: {train_rmse_va:.4f}% \n"
            f"Val Loss: {val_loss:.4f} | Val RMSE AR: {val_rmse_ar:.4f}% | Val RMSE VA: {val_rmse_va:.4f}% \n"
        )

        # EARLY STOPPING Y GUARDADO DEL MEJOR MODELO
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0

            # MJEOR MODELO
            model_save_path = "/content/drive/MyDrive/TFG/models/best_CNN_LSTM_emotions_reg.pth"
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "best_val_loss": best_val_loss}, model_save_path)
            print(f"Mejor modelo guardado en {model_save_path}")
        else:
            early_stop_counter += 1
            print(f"No mejora en validación por {early_stop_counter} época(s)")

        if early_stop_counter >= patience:
            print("Early stopping activado")
            break

    # GUARDAR EL MODELO FINAL
    final_model_save_path = "/content/drive/MyDrive/TFG/models/CNN_LSTM_emotions_reg.pth"
    torch.save(model.state_dict(), final_model_save_path)
    print(f"Modelo final guardado en {final_model_save_path}")

    
    val_labels_ar = np.array(val_labels_ar).squeeze()
    val_labels_va = np.array(val_labels_va).squeeze()
    val_preds_ar = np.array(val_preds_ar).squeeze()
    val_preds_va = np.array(val_preds_va).squeeze()

    print("Valores reales de valencia:", set(val_labels_va))  
    print("Valores reales de arousal:", set(val_labels_ar))
  
    #Scatter Plot:
    plot_scatter(val_labels_ar, val_preds_ar, "Predicciones vs. Valores reales (Arousal)", 
             "/content/drive/MyDrive/TFG/models/scatter_arousal.png")

    plot_scatter(val_labels_va, val_preds_va, "Predicciones vs. Valores reales (Valence)", 
             "/content/drive/MyDrive/TFG/models/scatter_valence.png")

    #Residual Plot
    plot_and_save_residuals(val_labels_ar, val_preds_ar, val_labels_va, val_preds_va, "/content/drive/MyDrive/TFG/models/residuals_emotions_reg.png")

    #Guardamos Metricas
    metrics_df = pd.DataFrame({
    'Epoch': epochs_list,
    'Train Loss': train_losses,
    'Val Loss': val_losses,
    'Val RMSE Arousal': rmse_arousal,
    'Val RMSE Valence': rmse_valence,
    'Val MAE Arousal': val_maes_ar,
    'Val MAE Valence': val_maes_va,
    'Val R2 Arousal': r2_arousal,
    'Val R2 Valence': r2_valence
})

    metrics_df.to_csv("/content/drive/MyDrive/TFG/models/training_metrics_emotions_reg.csv", index=False)

    # Convertir listas a numpy arrays
    val_labels_ar = np.array(all_labels_ar).squeeze()
    val_labels_va = np.array(all_labels_va).squeeze()
    val_preds_ar = np.array(all_preds_ar).squeeze()
    val_preds_va = np.array(all_preds_va).squeeze()
    val_probs_ar = np.array(val_probs_ar)
    val_probs_va = np.array(val_probs_va)

    # variables tienen la misma cantidad de muestras
    num_samples = val_labels_ar.shape[0]

    if (
        val_labels_va.shape[0] != num_samples or
        val_preds_ar.shape[0] != num_samples or
        val_preds_va.shape[0] != num_samples or
        val_probs_ar.shape[0] != num_samples or
        val_probs_va.shape[0] != num_samples
    ):
        raise ValueError("Las dimensiones de los arrays no coinciden. Revisa cómo se están acumulando los datos.")

    # Crear el DataFrame
    df_predictions = pd.DataFrame({
        'True Arousal': val_labels_ar,
        'Pred Arousal': val_preds_ar,
        'True Valence': val_labels_va,
        'Pred Valence': val_preds_va
    })

    # Guardar en CSV
    output_path = "/content/drive/MyDrive/TFG/models/predictions_emotions_probs_reg.csv"
    df_predictions.to_csv(output_path, index=False)
    print(f"Predicciones y probabilidades guardadas en {output_path}")


if __name__ == "__main__":
    main()
