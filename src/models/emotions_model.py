import torch
from torch import nn

class CNN_LSTM_emotions(nn.module):
    def __init__(self, num_classes, additional_features_dim, hidden_size):
        super(CNN_LSTM_emotions, self).__init__()

        # Bloque CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2))