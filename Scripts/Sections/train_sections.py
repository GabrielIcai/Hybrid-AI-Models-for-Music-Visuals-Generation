import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)


from src.models.structure_model import CRNN_Structure
from src.preprocessing.custom_dataset import CustomDataset_Sections
from src.preprocessing import (
    CustomDataset,
    load_data,
    normalize_columns,
    split_dataset,
    c_transform,
)
from src.training import collate_sections
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessing.data_loader import load_data, split_dataset
from src.training.trainer_sections import train_sections, validate_sections
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)

# Parametros
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
learning_rate = 0.001
weight_decay = 1e-5
data_path = "/content/drive/MyDrive/TFG/images/espectrogramas_normalizados_emociones_estructura/dataset_emociones_secciones.csv"
base_path = "/content/drive/MyDrive/TFG/images/"

mean=[0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std=[0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
hidden_size = 256
additional_features_dim = 10
num_classes = 5
learning_rate = 0.008
epochs = 50
weight_decay= 1e-4

def main():

    epochs_list = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_f1_scores = []
    val_precisions = []
    val_recalls = []
    data = load_data(data_path)
    data["Ruta"] = data["Ruta"].str.replace("\\", "/")
    data["Ruta"] = base_path + data["Ruta"]
    data["Ruta"] = data["Ruta"].str.replace("espectrogramas_salida_secciones_2", "espectrogramas_normalizados_emociones_estructura")
    label_columns = ["Break", "Pre-Drop", "Drop"]
    data = data.dropna(subset=label_columns)
    num_filas_antes = len(data)
    data = data[data[label_columns].sum(axis=1) == 1]
    num_filas_despues = len(data)
    print(f"Filas eliminadas por etiquetas inválidas: {num_filas_antes - num_filas_despues}")
    conteo_clases = data[label_columns].sum()
    print("Muestras por clase:")
    print(conteo_clases)
    normalize_columns(data, columns)
    
    for img_path in data["Ruta"]:
        if not os.path.exists(img_path):
            print(f"Ruta no encontrada: {img_path}")
    print("Rutas comprobadas")
    
    # Defino las Transformaciones
    train_transform = c_transform(mean, std)
    test_transform = c_transform(mean, std)

    print("Transformaciones definidas")

    # Train y Test
    train_data, test_data = split_dataset(data)
    print(f"Tamaño de train_data después de preprocesamiento: {len(train_data)}")
    print(f"Tamaño de test_data después de preprocesamiento: {len(test_data)}")

    # Mostrar el DataFrame completo
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Transformo los datos a tensores
    train_dataset = CustomDataset_Sections(train_data, base_path, transform=train_transform)
    test_dataset = CustomDataset_Sections(test_data, base_path, transform=test_transform)

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=128, collate_fn=collate_sections, shuffle=False, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        test_dataset, batch_size=128, collate_fn=collate_sections, shuffle=False, num_workers=2, pin_memory=True
    )
    print("DataLoaders creados")

    # Modelo
    model = CRNN_Structure(num_classes, additional_features_dim, hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #Early stopping
    patience = 5  
    best_val_loss = float("inf")
    early_stop_counter = 0

    # Entrenamiento
    for epoch in range(epochs):
    
        train_loss, train_accuracy = train_sections(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_accuracy, val_preds, val_labels, val_probs = validate_sections(
            model, val_loader, criterion, device
        )
        
        print(f"val_labels shape: {np.array(val_labels).shape}")
        print(f"val_labels: {val_labels}")
        print(f"val_preds shape: {np.array(val_preds).shape}")
        print(f"val_probs shape: {np.array(val_probs).shape}")

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average="weighted")
        val_recall = recall_score(val_labels, val_preds, average="weighted")
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        
        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f} | Val Precision: {val_precision:.4f} | "
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0

            # Guarda el mejor modelo
            model_save_path = "/content/drive/MyDrive/TFG/models/best_CRNN_sections.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Mejor modelo guardado en {model_save_path}")
        else:
            early_stop_counter += 1
            print(f"No mejora en validación por {early_stop_counter} época(s)")

        if early_stop_counter >= patience:
            print("Early stopping activado")
            break

    metrics_df = pd.DataFrame({
    'Epoch': epochs_list,
    'Train Loss': train_losses,
    'Val Loss': val_losses,
    'Train Accuracy': train_accuracies,
    'Val Accuracy': val_accuracies,
    'Val F1': val_f1_scores,
    'Val Precision': val_precisions,
    'Val Recall': val_recalls,
})
    #Guardo Métricas
    metrics_df.to_csv("/content/drive/MyDrive/TFG/models/training_metrics_sections_CRNN.csv", index=False)

    conf_matrix = confusion_matrix(val_labels, val_preds)

    # Mostrar y guardar como imagen
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Break", "Pre-Drop", "Drop"],
                yticklabels=["Break", "Pre-Drop", "Drop"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Final Confusion Matrix - Validation Set')
    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/TFG/models/final_confusion_matrix_secciones.png")
    plt.close()

    report = classification_report(val_labels, val_preds, target_names=["Break", "Pre-Drop", "Drop"])
    print(report)

# Guardar el reporte en un .txt si quieres
    with open("/content/drive/MyDrive/TFG/models/final_classification_report_secciones.txt", "w") as f:
        f.write(report)

    # Guardo el modelo
    model_save_path = "/content/drive/MyDrive/TFG/models/CRNN_sections.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Modelo guardado en {model_save_path}")


if __name__ == "__main__":
    main()
