import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(src_path)

print("Rutas en sys.path:")
for path in sys.path:
    print(path)

from src.preprocessing import (
    CustomDataset,
    c_transform,
    load_data,
    mean_std_image,
    split_dataset,
)


def load_image_from_csv(csv_path, base_path, index):

    data = pd.read_csv(csv_path)
    img_path = os.path.join(base_path, data.iloc[index]["Ruta"])

    if os.path.exists(img_path):
        try:
            # Cargar la imagen
            image = Image.open(img_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error al cargar la imagen {img_path}: {e}")
            return None
    else:
        print(f"Imagen no encontrada en la ruta: {img_path}")
        return None


base_path = "data//"
csv_path = "data/espectrogramas_salida1/dataset_genero_completo.csv"
index = 5

image = load_image_from_csv(csv_path, base_path, index)


columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
data_path = "data\espectrogramas_salida1\dataset_genero_completo.csv"
base_path = "data\\"

if image:
    image.show()
else:
    print("No se pudo cargar la imagen.")
