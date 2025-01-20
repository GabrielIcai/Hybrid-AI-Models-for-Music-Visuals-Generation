"""
This script contains the test
"""
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch
from src.models.genre_model import CNN_LSTM_genre
from src.preprocessing import normalize_columns, split_dataset, c_transform
from src.training import collate_fn, extract_song_name
import pandas as pd
from unittest.mock import patch
from src.preprocessing.custom_dataset import CustomDataset
from PIL import Image

def test_normalize_columns():
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    normalize_columns(data, ['col1'])
    print("Normalized DataFrame:")
    print(data)  # Esto mostrará el DataFrame después de la normalización
    assert data['col1'].min() == 0 and data['col1'].max() == 1
    assert data['col2'].min() == 4 and data['col2'].max() == 6

def test_split_dataset():
    data = pd.DataFrame({'col1': range(100)})
    train, test = split_dataset(data)
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print("Train indices:", train.index.tolist())
    print("Test indices:", test.index.tolist())
    assert len(train) + len(test) == 100
    assert len(set(train.index).intersection(set(test.index))) == 0

def test_custom_dataset():

    data_path = 'data/espectrogramas_salida1/dataset_genero_completo.csv'
    data = pd.read_csv(data_path)
    small_df = data.head(20)
    mean = [0.5, 0.5, 0.5] 
    std = [0.5, 0.5, 0.5]   
    transform = c_transform(mean, std)  
    base_path = "data\\"
    dataset = CustomDataset(small_df, base_path, transform)

    with patch("PIL.Image.open", return_value=Image.new("RGB", (155, 155))):
    
        assert len(dataset) == 20

        
        image, additional_features, labels = dataset[2]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 155, 155)
       
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (6,)  

    print("Características adicionales:", additional_features) 
    assert additional_features.shape == (12,)

def test_extract_song_name_with_real_data():
    data_path = 'data/espectrogramas_salida1/dataset_genero_completo.csv'
    data = pd.read_csv(data_path)
    small_df = data.head(5)
    image_paths = small_df['Ruta'].tolist()
    expected_song_names = [
        extract_song_name(path) for path in image_paths
    ]

    for path, expected_name in zip(image_paths, expected_song_names):
        assert extract_song_name(path) == expected_name, f"Error en {path}"

    print("Test de extract_song_name con datos reales pasó correctamente.")

def test_collate_fn_with_real_data():
    # Ruta al archivo CSV
    data_path = 'data/espectrogramas_salida1/dataset_genero_completo.csv'
    data = pd.read_csv(data_path)
    small_df = data.head(15)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = c_transform(mean, std)
    base_path = "data\\"
    dataset = CustomDataset(small_df, base_path, transform)

    # Creo un lote simulando un DataLoader
    with patch("PIL.Image.open", return_value=Image.new("RGB", (155, 155))):
        batch = [
            (dataset[i][0], dataset[i][1], dataset[i][2], f"song{i // 3}_fragmento_{i}.png")
            for i in range(len(dataset))
        ]

    images, additional_features, labels = collate_fn(batch)

    assert images.shape == (5, 3, 3, 155, 155)  # 5 canciones, 3 fragmentos por canción, canales y dimensiones
    assert additional_features.shape == (5, 3, 12)  # 5 canciones, 3 fragmentos por canción, 12 características
    assert labels.shape == (5, 3, 6)  # 5 canciones, 3 fragmentos por canción, 6 etiquetas

    print("Test de collate_fn con datos reales pasó correctamente.")