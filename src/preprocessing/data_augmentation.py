import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import pandas as pd

def mel_spectrogram_augmentation(spectrogram):
    spectrogram = Image.fromarray(spectrogram)

    transform = transforms.Compose([
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)) ])

    augmented_spectrogram = transform(spectrogram)
    augmented_spectrogram = np.array(augmented_spectrogram)
    
    return augmented_spectrogram

def augment_dataframe(df, columns_to_augment):
    augmented_data = [] 
    
    for index, row in df.iterrows():
        augmented_row = row.copy()
        for col in columns_to_augment:
            augmented_row[col] = mel_spectrogram_augmentation(row[col])
        augmented_data.append(augmented_row)
    
    augmented_df = pd.DataFrame(augmented_data, index=df.index)
    
    return augmented_df