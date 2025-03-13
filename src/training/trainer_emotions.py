import torch
import numpy as np


def trainer_emotions(model, train_loader, optimizer, criterion, device):
    model.train()
    
    running_loss = 0.0
    mse_ar = 0.0
    mse_va = 0.0
    total = 0

    for i, batch in enumerate(train_loader):
        images, additional_features, valencia_labels, arousal_labels = batch
        images, additional_features, valencia_labels, arousal_labels = images.to(device), additional_features.to(device), valencia_labels.to(device), arousal_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        val_output, ar_output = model(images, additional_features)

        # Cálculo de pérdida
        perdida_ar = criterion(ar_output, arousal_labels)
        perdida_va = criterion(val_output, valencia_labels)

        loss = perdida_ar + perdida_va

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #MSE
        mse_ar += perdida_ar.item()
        mse_va += perdida_va.item()
        total += 1

        print(f"MSE Arousal: {mse_ar:.4f} | MSE Valence: {mse_va:.4f}")

    avg_mse_ar = mse_ar / total
    avg_mse_va = mse_va / total
    avg_rmse_ar = np.sqrt(avg_mse_ar)
    avg_rmse_va = np.sqrt(avg_mse_va)

    return running_loss / len(train_loader), avg_rmse_ar, avg_rmse_va

def validate_emotions(model, val_loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    mse_ar = 0
    mse_va = 0
    total = 0

    val_preds_ar = []
    val_preds_va = []
    val_labels_va = []
    val_labels_ar = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images, additional_features, valencia_labels, arousal_labels = batch
            images, additional_features, valencia_labels, arousal_labels = (
                images.to(device),
                additional_features.to(device),
                valencia_labels.to(device),
                arousal_labels.to(device),
            )

            # Forward pass
            val_output, ar_output = model(images, additional_features)

            # Cálculo de pérdida (corregido)
            perdida_ar = criterion(ar_output, arousal_labels)
            perdida_va = criterion(val_output, valencia_labels)
            loss = perdida_ar + perdida_va

            running_loss += loss.item()
            
            #Guardamos los mse
            mse_ar += perdida_ar.item()
            mse_va += perdida_va.item()
            total += 1

            # Guardamos las predicciones
            val_preds_ar.extend(ar_output.cpu().numpy())
            val_preds_va.extend(val_output.cpu().numpy())

            # Guardamos las etiquetas correctas
            val_labels_va.extend(valencia_labels.cpu().numpy())
            val_labels_ar.extend(arousal_labels.cpu().numpy())

            print(f"Shape de ar_output: {ar_output.shape}")
            print(f"Shape de val_output: {val_output.shape}")
            print(f"RMSE Arousal: {mse_ar:.4f} | MSE Valence: {mse_va:.4f}")
            print(f"Shape de ar_output: {ar_output.shape}")
            print(f"Shape de val_output: {val_output.shape}")
            print(f"RMSE Arousal: {mse_ar:.4f} | MSE Valence: {mse_va:.4f}")

    avg_mse_ar = mse_ar / total
    avg_mse_va = mse_va / total
    avg_rmse_ar = np.sqrt(avg_mse_ar)
    avg_rmse_va = np.sqrt(avg_mse_va)

    return (running_loss / len(val_loader), avg_rmse_ar, avg_rmse_va, 
        val_preds_ar, val_preds_va, 
        np.array(val_labels_ar), np.array(val_labels_va))