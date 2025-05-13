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
        batch_size, channels, height, width = x.size()
        print(f"Entrada inicial: {x.shape}")
        x = self.resnet(x) 
        x = x.view(batch_size, -1)
        x = torch.cat((x, additional_features), dim=-1)
        x = x.unsqueeze(1)  
    
        x, _ = self.rnn(x)
        out = self.fc(x[:, -1, :])
        print(f"Salida final: {out.shape}")
        return out

        