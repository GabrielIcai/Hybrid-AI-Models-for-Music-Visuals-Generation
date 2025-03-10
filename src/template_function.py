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

