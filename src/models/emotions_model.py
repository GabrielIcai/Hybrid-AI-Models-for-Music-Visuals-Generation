import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCRNNEmotionModel(nn.Module):
    def __init__(self, num_valencia_classes=10, num_arousal_classes=10, hidden_size=256, num_layers=2):
        super(ResNetCRNNEmotionModel, self).__init__()
        
        #RESNET-18
        resnet = models.resnet18(pretrained=True)
        
        #Ultima capa de resnet no nos vale
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        #LSTM
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        #Necesito dos salidas: valencia y arousal
        self.fc_valencia = nn.Linear(hidden_size * 2, num_valencia_classes)
        self.fc_arousal = nn.Linear(hidden_size * 2, num_arousal_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), 1, -1)
        lstm_out, _ = self.lstm(x)
        
        lstm_out = lstm_out[:, -1, :]
        valencia_output = self.fc_valencia(lstm_out)
        arousal_output = self.fc_arousal(lstm_out)
        
        return valencia_output, arousal_output
