import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN_LSTM_genre(nn.Module):
    def __init__(self, num_classes, additional_features_dim, hidden_size):
        super(CNN_LSTM_genre, self).__init__()

        # Bloque CNN
        self.cnn = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # (128x128)-> (64x64)
            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # (64x64) -> (32x32)
            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # (32x32) -> (16x16)
        )

        # Bloque LSTM
        self.lstm = nn.LSTM(
            input_size=16 * 16 * 128 + additional_features_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )

        # Capa Final
        self.fc = nn.Linear(256, num_classes)


    def forward(self, x, additional_features):

        batch_size, seq_len, channels, height, width = x.size()

        # Primero aplicamos la CNN a cada fragmento de una secuencia de tres:
        print(f"Forma de x antes de la CNN: {x.size()}")
        x = x.view(seq_len * batch_size, channels, height, width)
        x = self.cnn(x)  # Bloque Cnn
        x = x.view(batch_size, seq_len, -1)
        print(f"Forma de x después de la CNN: {x.size()}")
        # Añado las caracteristicas adicionales
        x = torch.cat((x, additional_features), dim=-1)
        print(f"Forma de x después de concatenar características adicionales: {x.size()}")
        # LSTM
        lstm_out, _ = self.lstm(x)
        # Capa final
        out = self.fc(lstm_out[:, -1, :])
        return out
