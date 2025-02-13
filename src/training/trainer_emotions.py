import torch

import torch

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    
    running_loss = 0.0
    correct_ar = 0
    correct_va = 0
    total = 0

    for i, batch in enumerate(train_loader):
        images, additional_features, labels = batch
        images = images.to(device)
        additional_features = additional_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        # Forward
        val_output, ar_output = model(images, additional_features)

        #Convierto a hot-encoding
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)

        # Cálculo de pérdidas
        perdida_ar = criterion(ar_output, labels)
        perdida_va = criterion(val_output, labels)
        loss = perdida_ar + perdida_va  # Sumar las pérdidas para un solo backward
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Predicciones
        _, predicted_ar = torch.max(ar_output.data, 1)
        _, predicted_va = torch.max(val_output.data, 1)

        correct_ar += (predicted_ar == labels).sum().item()
        correct_va += (predicted_va == labels).sum().item()
        total += labels.size(0)

    accuracy_ar = 100 * correct_ar / total
    accuracy_va = 100 * correct_va / total

    return running_loss / len(train_loader), accuracy_ar, accuracy_va

