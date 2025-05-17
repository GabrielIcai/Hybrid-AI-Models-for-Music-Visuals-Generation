import torch

def train_sections(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images, additional_features, labels = batch
        images = images.to(device)
        additional_features = additional_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, additional_features)

        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)

        labels = torch.argmax(labels, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate_sections(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            images, additional_features, labels = batch
            images = images.to(device)
            additional_features = additional_features.to(device)
            labels = labels.to(device)

            outputs = model(images, additional_features)
            labels = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels, all_probs
