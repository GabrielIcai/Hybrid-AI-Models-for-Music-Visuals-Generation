import torch


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        images, additional_features, labels = batch
        images = images.to(device)
        additional_features = additional_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, additional_features)
        print("labels before conversion")
        print(labels)

        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)  # Convertir a índices
            print(f"Labels (after conversion to indices): {labels}")
            print(f"Labels shape (after conversion): {labels.size()}")
            
        #Comprobaciones de GPU
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0)) 

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Predicciones
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = 100 * correct / total

    return running_loss / len(train_loader), accuracy


def validate(model, test_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    correct = 0
    total = 0

    with torch.no_grad():  # No se calculan los gradientes, es decir no se actualizan los pesos.
        for batch in test_loader:
            images, additional_features, labels = batch
            images = images.to(device)
            additional_features = additional_features.to(device)
            labels = labels.to(device)
            
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)
            
            outputs = model(images, additional_features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            #Probabilidades con softmax:
            probs = torch.softmax(outputs, dim=1)
            # Predicciones
            preds = torch.argmax(outputs, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            val_accuracy = 100 * correct / total if total > 0 else 0

    return val_loss / len(test_loader), val_accuracy, val_preds, val_labels

