import os
import sys
import numpy as np
import pandas as pd
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocessing.custom_dataset import EmotionDataset
from src.training.trainer_emotions import trainer_emotions, validate_emotions
import seaborn as sns
import matplotlib.pyplot as plt

mean=[0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std=[0.21755781769752502, 0.15407244861125946, 0.07557372003793716]

# Cargar datos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
learning_rate = 0.001
weight_decay = 1e-4
data_path = "/content/drive/MyDrive/TFG/images/espectrogramas_normalizados_emociones_estructura/dataset_emociones_secciones.csv"
base_path = "/content/drive/MyDrive/TFG/images/"


def main():
    epochs = 50
    patience = 5
    best_val_loss = float("inf")
    early_stop_counter = 0

    epochs_list = []
    train_losses, val_losses = [], []
    train_accuracies_ar, val_accuracies_ar = [], []
    train_accuracies_va, val_accuracies_va = [], []
    val_f1_scores_ar, val_f1_scores_va = [], []
    val_precisions_ar, val_precisions_va = [], []
    val_recalls_ar, val_recalls_va = [], []
    all_preds_ar = []
    all_preds_va = []
    all_labels_ar = []
    all_labels_va = []

    data = load_data(data_path)
    data["Ruta"] = data["Ruta"].str.replace("\\", "/")
    data["Ruta"] = base_path + data["Ruta"]
    data["Ruta"] = data["Ruta"].str.replace("espectrogramas_salida_secciones_2", "espectrogramas_normalizados_emociones_estructura")

    print("Primeras filas del dataset:")
    data=data.head(30)
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        # ENTRENAMIENTO
        train_loss, train_acc_ar, train_acc_va = trainer_emotions(
            model, train_loader, optimizer, criterion, device
        )

        # VALIDACIÓN
        val_loss, val_acc_ar, val_acc_va, val_preds_ar, val_preds_va, val_labels_ar, val_labels_va, val_probs_ar, val_probs_va = validate_emotions(
            model, val_loader, criterion, device
        )

        all_preds_ar.extend(val_preds_ar)
        all_preds_va.extend(val_preds_va)

        # ETIQUETAS A ÍNDICES
        val_labels_ar = np.array(val_labels_ar)
        val_labels_va = np.array(val_labels_va)

        if val_labels_ar.ndim > 1:
            val_labels_ar = val_labels_ar.argmax(axis=1)

        if val_labels_va.ndim > 1:
            val_labels_va = val_labels_va.argmax(axis=1)
        
        all_labels_ar.extend(val_labels_ar) 
        all_labels_va.extend(val_labels_va)

        # MATRICES DE CONFUSIÓN
        cm_arousal = confusion_matrix(val_labels_ar, val_preds_ar)
        cm_valence = confusion_matrix(val_labels_va, val_preds_va)

        # Guardar matrices en CSV con nombres de columnas y filas
        np.savetxt(f"/content/drive/MyDrive/TFG/models/confusion_matrix_arousal_epoch_{epoch+1}.csv", cm_arousal, delimiter=",", fmt="%d")
        np.savetxt(f"/content/drive/MyDrive/TFG/models/confusion_matrix_valencia_epoch_{epoch+1}.csv", cm_valence, delimiter=",", fmt="%d")
        print(f"Matrices de confusión guardadas para la época {epoch+1}.")

        # Guardar métricas de la época
        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies_ar.append(train_acc_ar)
        train_accuracies_va.append(train_acc_va)
        val_accuracies_ar.append(val_acc_ar)
        val_accuracies_va.append(val_acc_va)

        # Calcular métricas adicionales
        val_f1_ar = f1_score(val_labels_ar, val_preds_ar, average="weighted")
        val_f1_va = f1_score(val_labels_va, val_preds_va, average="weighted")
        val_precision_ar = precision_score(val_labels_ar, val_preds_ar, average="weighted")
        val_precision_va = precision_score(val_labels_va, val_preds_va, average="weighted")
        val_recall_ar = recall_score(val_labels_ar, val_preds_ar, average="weighted")
        val_recall_va = recall_score(val_labels_va, val_preds_va, average="weighted")

        # Guardar métricas en listas
        val_f1_scores_ar.append(val_f1_ar)
        val_f1_scores_va.append(val_f1_va)
        val_precisions_ar.append(val_precision_ar)
        val_precisions_va.append(val_precision_va)
        val_recalls_ar.append(val_recall_ar)
        val_recalls_va.append(val_recall_va)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc AR: {train_acc_ar:.4f} | Train Acc VA: {train_acc_va:.4f} \n"
            f"Val Loss: {val_loss:.4f} | Val Acc AR: {val_acc_ar:.4f} | Val Acc VA: {val_acc_va:.4f} \n"
        )

        # EARLY STOPPING Y GUARDADO DEL MEJOR MODELO
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0

            # MJEOR MODELO
            model_save_path = "/content/drive/MyDrive/TFG/models/best_CNN_LSTM_emotions.pth"
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
    final_model_save_path = "/content/drive/MyDrive/TFG/models/CNN_LSTM_emotions.pth"
    torch.save(model.state_dict(), final_model_save_path)
    print(f"Modelo final guardado en {final_model_save_path}")

    #MATRICES EN CSV
    cm_arousal = confusion_matrix(all_labels_ar, all_preds_ar)
    cm_valence = confusion_matrix(all_labels_va, all_preds_va)

    # Guardar matrices en CSV
    np.savetxt("/content/drive/MyDrive/TFG/models/confusion_matrix_arousal_final.csv", cm_arousal, delimiter=",", fmt="%d")
    np.savetxt("/content/drive/MyDrive/TFG/models/confusion_matrix_valence_final.csv", cm_valence, delimiter=",", fmt="%d")
    print("Matrices de confusión finales guardadas.")

    metrics_df = pd.DataFrame({
    'Epoch': epochs_list,
    'Train Loss': train_losses,
    'Val Loss': val_losses,
    'Train Accuracy Arousal': train_accuracies_ar,
    'Train Accuracy Valence': train_accuracies_va,
    'Val Accuracy Arousal': val_accuracies_ar,
    'Val Accuracy Valence': val_accuracies_va,
    'Val F1 Arousal': val_f1_scores_ar,
    'Val F1 Valence': val_f1_scores_va,
    'Val Precision Arousal': val_precisions_ar,
    'Val Precision Valence': val_precisions_va,
    'Val Recall Arousal': val_recalls_ar,
    'Val Recall Valence': val_recalls_va,
})
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_arousal, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicciones")
    plt.ylabel("Valores reales")
    plt.title("Matriz de Confusión - Arousal")
    plt.savefig("/content/drive/MyDrive/TFG/models/confusion_matrix_arousal_final.png")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_valence, annot=True, fmt="d", cmap="Oranges")
    plt.xlabel("Predicciones")
    plt.ylabel("Valores reales")
    plt.title("Matriz de Confusión - Valence")
    plt.savefig("/content/drive/MyDrive/TFG/models/confusion_matrix_valence_final.png")

    plt.show()

if __name__ == "__main__":
    main()
