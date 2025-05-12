import torch
import torch.nn as nn
import torchvision.models as models

class CRNN_Structure(nn.Module):
    def __init__(self, num_classes, additional_features_dim, hidden_size):
        super(CRNN_Structure, self).__init__()
        
        # Bloque CNN
        resnet = models.resnet18(pretrained=True)
        
        #Ultima capa de resnet no nos vale (ya que utilizamos GRU al final)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        
        # Bloque RNN
        self.rnn = nn.GRU(
            input_size=128 * 8 * 8 +additional_features_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, additional_features):
        batch_size, seq_len, channels, height, width = x.size()  # x: (batch_size, 3, 1, 128, 128)
        print(f"Entrada inicial: {x.shape}")

        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        print(f"Salida de la CNN: {x.shape}")
        print(f"Características adicionales: {additional_features.shape}")
        x = torch.cat((x, additional_features), dim=-1)
        x, _ = self.rnn(x)
        out = self.fc(x[:, -1, :])
        print(f"Salida final: {out.shape}")
        return out
        