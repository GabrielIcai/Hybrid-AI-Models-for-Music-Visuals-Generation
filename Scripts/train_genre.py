import os
import sys

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)


from src.models.genre_model import CNN_LSTM_genre
from src.preprocessing import (
    CustomDataset,
    load_data,
    mean_std_image,
    normalize_columns,
    normalize_images,
    split_dataset,
    c_transform,
)
from src.training import collate_fn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessing.data_loader import load_data, split_dataset
from src.preprocessing.custom_dataset import CustomDataset
from src.training.trainer_genre import train, validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
# Parametros
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
#Cargo las imagenes desde drive
data_path = "/content/drive/MyDrive/TFG/images/espectrogramas_normalizados/dataset_genero_completo.csv"
base_path = "/content/drive/MyDrive/TFG/images/"
hidden_size = 256
additional_features_dim = 12
num_classes = 6
learning_rate = 0.001
epochs = 50


def main():
     # Preprocesado
    data = load_data(data_path)
    data["Ruta"] = data["Ruta"].str.replace("\\", "/")
    data["Ruta"] = base_path + data["Ruta"]
    data["Ruta"] = data["Ruta"].str.replace("espectrogramas_salida1", "espectrogramas_normalizados")

    print(data.head(10))
    data = data.head(20)

    normalize_columns(data, columns)

    print(data.head(4))

    for img_path in data["Ruta"]:
        if not os.path.exists(img_path):
            print(f"Ruta no encontrada: {img_path}")
    print("Rutas comprobadas")

    mean, std = mean_std_image(data)
    
    print(mean,std)

    # Defino las Transformaciones
    train_transform = c_transform(mean, std)
    test_transform = c_transform(mean, std)

    print("Transformaciones definidas")

    # Train y Test
    train_data, test_data = split_dataset(data)
    print(f"Tamaño de train_data después de preprocesamiento: {len(train_data)}")
    print(f"Tamaño de test_data después de preprocesamiento: {len(test_data)}")
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Transformo los datos a tensores
    train_dataset = CustomDataset(train_data, base_path, transform=train_transform)
    test_dataset = CustomDataset(test_data, base_path, transform=test_transform)
    print("Datos transformados a tensores")
    print(train_dataset.head(), test_dataset.head())

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False
    )
    val_loader = DataLoader(
        test_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False
    )
    print("DataLoaders creados")

    # Modelo
    model = CNN_LSTM_genre(num_classes, additional_features_dim, hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento
    for epoch in range(epochs):
        train_loss, train_accuracy = train(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_accuracy, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        val_precision = precision_score(val_labels, val_preds, average="weighted")
        val_recall = recall_score(val_labels, val_preds, average="weighted")
        val_auc = roc_auc_score(
            val_labels, val_preds, average="weighted", multi_class="ovr"
        )

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f} | Val Precision: {val_precision:.4f} | "
            f"Val Recall: {val_recall:.4f} | Val AUC: {val_auc:.4f}"
        )

        with open("metrics.txt", "a") as f:
            f.write(
                f"Epoch {epoch + 1}: "
                f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | "
                f"Val F1: {val_f1:.4f} | Val Precision: {val_precision:.4f} | "
                f"Val Recall: {val_recall:.4f} | Val AUC: {val_auc:.4f}\n"
            )

    # Guardar el modelo
    model_save_path = "/content/TFG/src/models/cnn_lstm_genre.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Modelo guardado en {model_save_path}")


if __name__ == "__main__":
    main()
