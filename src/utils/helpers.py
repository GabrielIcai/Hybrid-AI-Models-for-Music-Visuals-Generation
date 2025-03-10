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



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_scatter(y_true, y_pred, title, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predicciones")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label="Ideal (y = x)")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(save_path)

def plot_and_save_residuals(val_labels_ar, val_preds_ar, val_labels_va, val_preds_va, save_path):

    val_preds_ar = np.array(val_preds_ar).squeeze()
    val_preds_va = np.array(val_preds_va).squeeze()
    
    # Calcular los residuos
    residuals_ar = val_labels_ar - val_preds_ar
    residuals_va = val_labels_va - val_preds_va 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(x=val_preds_ar, y=residuals_ar, alpha=0.5, ax=axes[0])
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicciones Arousal')
    axes[0].set_ylabel('Residuos (Error)')
    axes[0].set_title('Residual Plot - Arousal')

    sns.scatterplot(x=val_preds_va, y=residuals_va, alpha=0.5, ax=axes[1])
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_xlabel('Predicciones Valencia')
    axes[1].set_ylabel('Residuos (Error)')
    axes[1].set_title('Residual Plot - Valencia')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gráfico de residuos guardado en: {save_path}")

