import torch

def accuracy_with_tolerance(y_pred, y_true, tolerance=1):
    print("TAMAÑO DE Y_TRUE")
    print(y_true.shape)
    pred_index = torch.argmax(y_pred, dim=1)
    true_index = y_true
    
    val_range = torch.linspace(0, 1, steps=11, device=y_pred.device)
    pred_values = val_range[pred_index]
    true_values = val_range[true_index]

    correct = (torch.abs(pred_values - true_values) <= 0.1).float()
    accuracy = correct.mean().item() * 100
    
    return accuracy


def trainer_emotions(model, train_loader, optimizer, criterion, device):
    model.train()
    
    running_loss = 0.0
    correct_ar = 0
    correct_va = 0
    total = 0

    for i, batch in enumerate(train_loader):
        images, additional_features, valencia_labels, arousal_labels = batch
        images, additional_features, valencia_labels, arousal_labels = images.to(device), additional_features.to(device), valencia_labels.to(device), arousal_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        val_output, ar_output = model(images, additional_features)

        if valencia_labels.dim() > 1:
            valencia_labels = torch.argmax(valencia_labels, dim=1)
        if arousal_labels.dim() > 1:
            arousal_labels = torch.argmax(arousal_labels, dim=1)


        # Cálculo de pérdida
        perdida_ar = criterion(ar_output, arousal_labels)
        perdida_va = criterion(val_output, valencia_labels)

        loss = perdida_ar + perdida_va

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # TOLERANCIA
        accuracy_ar = accuracy_with_tolerance(ar_output, arousal_labels)
        accuracy_va = accuracy_with_tolerance(val_output, valencia_labels)

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

            # One-hot encoding handling
            if valencia_labels.dim() > 1:
                valencia_labels = torch.argmax(valencia_labels, dim=1)
            if arousal_labels.dim() > 1:
                arousal_labels = torch.argmax(arousal_labels, dim=1)

            # Cálculo de pérdida (corregido)
            perdida_ar = criterion(ar_output, arousal_labels)
            perdida_va = criterion(val_output, valencia_labels)
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

            # Guardamos las etiquetas correctas
            val_labels_va.extend(valencia_labels.cpu().numpy())
            val_labels_ar.extend(arousal_labels.cpu().numpy())

            # Cálculo de accuracy con tolerancia (corregido)
            accuracy_ar = accuracy_with_tolerance(ar_output, arousal_labels)
            accuracy_va = accuracy_with_tolerance(val_output, valencia_labels)

            correct_ar += accuracy_ar
            correct_va += accuracy_va
            total += 1

    accuracy_ar = 100 * correct_ar / total
    accuracy_va = 100 * correct_va / total

    return (running_loss / len(val_loader), accuracy_ar, accuracy_va, 
            val_preds_ar, val_preds_va, val_labels_ar, val_labels_va, val_probs_ar, val_probs_va)

