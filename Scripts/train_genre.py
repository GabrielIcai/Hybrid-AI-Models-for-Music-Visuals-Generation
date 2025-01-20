src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(src_path)

import sys
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
import os
from src.training.trainer_genre import train, validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


from src.preprocessing import (
    CustomDataset,
    c_transform,
    load_data,
    mean_std_image,
    split_dataset,
)

# Parametros
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
data_path = "data\espectrogramas_salida1\dataset_genero_completo.csv"
base_path = "data\\"
dir_out = "drive/MyDrive"
hidden_size = 256
additional_features_dim = 12
num_classes = 6
learning_rate = 0.001
epochs = 50


def main():
    # Preprocesado
    data = load_data(data_path)
    normalize_columns(data, columns)
    normalize_images(data, base_path)
    mean, std = mean_std_image(data)

    # Defino las Transformaciones
    train_transform = c_transform(mean, std)
    test_transform = c_transform(mean, std)

    # Train y Test
    train_data, test_data = split_dataset(data)

    # Transformo los datos a tensores
    train_dataset = CustomDataset(train_data, base_path, transform=train_transform)
    test_dataset = CustomDataset(test_data, base_path, transform=test_transform)

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=False
    )
    val_loader = DataLoader(
        test_dataset, batch_size=16, collate_fn=collate_fn, shuffle=False
    )

    os.makedirs(dir_out, exist_ok=True)

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
