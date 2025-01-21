import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2
import os
from torchvision import transforms
from PIL import Image


def normalize_columns(data, columns):
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])


def normalize_images(data, base_path):
    normalized_images = []
    for img_path in data["Ruta"]:
        try:
            img = cv2.imread(os.path.join(base_path, img_path))
            img = img / 255.0
            normalized_images.append(img)
        except Exception as e:
            print(f"Error normalizing image {img_path}: {e}")
    return np.array(normalized_images)


# Para calcular la media y varianza y luego aplicarlas en el tranforms
def mean_std_image(data):
    mean = 0.0
    std = 0.0
    n_s = 0
    transform = transforms.ToTensor()

    for idx, row in data.iterrows():
        img_path = + row["Ruta"]
        try:
            image = Image.open(img_path).convert("RGB")
            batch_s = 1
            n_s += batch_s
            tensor_image = transform(image)
            mean += tensor_image.mean([1, 2])
            std += tensor_image.std([1, 2])
        except Exception as e:
            print(f"Error {img_path}: {e}")
    mean = mean / n_s
    std = std / n_s
    return mean, std
