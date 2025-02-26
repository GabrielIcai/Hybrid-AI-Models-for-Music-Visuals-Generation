import torch

def accuracy_with_tolerance(y_pred, y_true, tolerance=1):
    pred_index = torch.argmax(y_pred, dim=1)
    true_index = torch.argmax(y_true, dim=1)
    
    val_range = torch.linspace(0, 1, steps=11, device=y_pred.device)
    pred_values = val_range[pred_index]
    true_values = val_range[true_index]

    correct = (torch.abs(pred_values - true_values) <= 0.1).float()
    accuracy = correct.mean().item() * 100
    
    return accuracy


def train_emotions(model, train_loader, optimizer, criterion, device):
    model.train()
    
    running_loss = 0.0
    correct_ar = 0
    correct_va = 0
    total = 0

    for i, batch in enumerate(train_loader):
        images, additional_features, labels = batch
        images, additional_features, labels = images.to(device), additional_features.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        val_output, ar_output = model(images, additional_features)

        #One-hot encoding
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)

        # Cálculo de pérdida
        perdida_ar = criterion(ar_output, labels)
        perdida_va = criterion(val_output, labels)
        loss = perdida_ar + perdida_va

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # TOLERANCIA
        accuracy_ar = accuracy_with_tolerance(ar_output, labels)
        accuracy_va = accuracy_with_tolerance(val_output, labels)

        correct_ar += accuracy_ar
        correct_va += accuracy_va
        total += 1

    accuracy_ar = 100 * correct_ar / total
    accuracy_va = 100 * correct_va / total

    return running_loss / len(train_loader), accuracy_ar, accuracy_va

def validate_emotions(model, val_loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    correct_ar = 0
    correct_va = 0
    total = 0

    val_preds_ar = []
    val_preds_va = []
    val_probs_ar = []
    val_probs_va = []
    val_labels = []

    with torch.no_grad():  # Desactivamos gradientes para validar más rápido y ahorrar memoria
        for i, batch in enumerate(val_loader):
            images, additional_features, labels = batch
            images, additional_features, labels = images.to(device), additional_features.to(device), labels.to(device)

            # Forward pass
            val_output, ar_output = model(images, additional_features)

            # One-hot encoding
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)

            # Cálculo de pérdida
            perdida_ar = criterion(ar_output, labels)
            perdida_va = criterion(val_output, labels)
            loss = perdida_ar + perdida_va

            running_loss += loss.item()

            # Probabilidades y predicciones
            probs_ar = torch.softmax(ar_output, dim=1)
            probs_va = torch.softmax(val_output, dim=1)
            preds_ar = torch.argmax(probs_ar, dim=1)
            preds_va = torch.argmax(probs_va, dim=1)

            val_probs_ar.extend(probs_ar.cpu().numpy())
            val_probs_va.extend(probs_va.cpu().numpy())
            val_preds_ar.extend(preds_ar.cpu().numpy())
            val_preds_va.extend(preds_va.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

            # Cálculo de accuracy con tolerancia
            accuracy_ar = accuracy_with_tolerance(ar_output, labels)
            accuracy_va = accuracy_with_tolerance(val_output, labels)

            correct_ar += accuracy_ar
            correct_va += accuracy_va
            total += 1

    accuracy_ar = 100*correct_ar / total
    accuracy_va = 100*correct_va / total

    return (running_loss / len(val_loader), accuracy_ar, accuracy_va, 
            val_preds_ar, val_preds_va, val_labels, val_probs_ar, val_probs_va)

