import torch


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct=0
    total=0

    for batch in train_loader:
        images, additional_features, labels = batch
        images = images.to(device)
        additional_features = additional_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, additional_features)
        print(f"Forma de labels: {labels.size()}")
        #Calculo perdida
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        _, labels_max = torch.max(labels, 1)
        correct += (predicted == labels_max).sum().item()
        total += labels.size(0)
        accuracy = 100 * correct / total

    return running_loss / len(train_loader), accuracy


def validate(model, test_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []

    with torch.no_grad():  # No se calculan los gradientes, es decir no se actualizan los pesos.
        for batch in test_loader:
            images, additional_features, labels = batch
            images = images.to(device)
            additional_features = additional_features.to(device)
            labels = labels.to(device)

            outputs = model(images, additional_features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Predicciones
            preds = torch.argmax(outputs, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    return val_loss / len(test_loader), val_preds, val_labels
