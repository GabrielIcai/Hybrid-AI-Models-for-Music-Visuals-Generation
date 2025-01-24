import torch
import torch.nn as nn
import torchvision.models as models

class EmotionRecognitionCNN_LSTM(nn.Module):
    def __init__(self, num_classes, additional_features_dim, hidden_size):
        super(EmotionRecognitionCNN_LSTM, self).__init__()
        
        #CNN
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512 + additional_features_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, additional_features):
        cnn_features = self.resnet(x)  # (batch_size, 512)
        
        combined_features = torch.cat((cnn_features, additional_features), dim=1)
        lstm_input = combined_features.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        output = self.fc(lstm_out)
        
        return output
