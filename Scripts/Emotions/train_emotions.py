import os
import sys
import numpy as np
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
from src.training.trainer_emotions import train_emotions, validate_emotions
import seaborn as sns
import matplotlib.pyplot as plt

mean=[0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std=[0.21755781769752502, 0.15407244861125946, 0.07557372003793716]

# Cargar datos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        train_loss, train_acc_ar, train_acc_va = train_emotions(
            model, train_loader, optimizer, criterion, device
        )

        # VALIDACIÓN
        val_loss, val_acc_ar, val_acc_va, val_preds_ar, val_preds_va, val_labels, _, _ = validate_emotions(
            model, val_loader, criterion, device
        )

        # ETIQUETAS A INDICES
        val_labels_idx = val_labels.argmax(axis=1)

        # MATRICES DE CONFUSIÓN
        cm_arousal = confusion_matrix(val_labels_idx, val_preds_ar)
        cm_valence = confusion_matrix(val_labels_idx, val_preds_va)

        # Guardar matrices en CSV con nombres de columnas y filas
        np.savetxt("/content/drive/MyDrive/TFG/models/confusion_matrix_arousal.csv", cm_arousal, delimiter=",", fmt="%d")
        np.savetxt("/content/drive/MyDrive/TFG/models/confusion_matrix_valencia.csv", cm_valence, delimiter=",", fmt="%d")
        print("Matrices de confusión guardadas.")

        # Guardar métricas de la época
        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies_ar.append(train_acc_ar)
        train_accuracies_va.append(train_acc_va)
        val_accuracies_ar.append(val_acc_ar)
        val_accuracies_va.append(val_acc_va)

        # Calcular métricas adicionales
        val_f1_ar = f1_score(val_labels_idx, val_preds_ar, average="weighted")
        val_f1_va = f1_score(val_labels_idx, val_preds_va, average="weighted")
        val_precision_ar = precision_score(val_labels_idx, val_preds_ar, average="weighted")
        val_precision_va = precision_score(val_labels_idx, val_preds_va, average="weighted")
        val_recall_ar = recall_score(val_labels_idx, val_preds_ar, average="weighted")
        val_recall_va = recall_score(val_labels_idx, val_preds_va, average="weighted")

        val_f1_scores_ar.append(val_f1_ar)
        val_f1_scores_va.append(val_f1_va)
        val_precisions_ar.append(val_precision_ar)
        val_precisions_va.append(val_precision_va)
        val_recalls_ar.append(val_recall_ar)
        val_recalls_va.append(val_recall_va)

        # Visualizar matrices de confusión
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.heatmap(cm_arousal, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores reales")
        plt.title("Matriz de Confusión - Arousal")

        plt.subplot(1, 2, 2)
        sns.heatmap(cm_valence, annot=True, fmt="d", cmap="Oranges")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores reales")
        plt.title("Matriz de Confusión - Valence")

        plt.show()

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


if __name__ == "__main__":
    main()
