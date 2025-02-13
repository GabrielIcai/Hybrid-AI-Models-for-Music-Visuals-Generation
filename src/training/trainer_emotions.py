import torch

def train(model, train_loader, optimizer, criterion, device):
    model.train()


    for i, batch in enumerate(train_loader):
        images, addtional_features, labels = batch
        images = images.to(device)
        addtional_features = addtional_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        val_output, ar_output = model(images, addtional_features)

        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)

        perdida_ar=criterion(ar_output, labels)
        perdida_va=criterion(val_output, labels)
        perdida_ar.backward()
        perdida_va.backward()
        optimizer.step()
        running_loss += perdida_va.item()
        running_loss += perdida_ar.item()

        #Predicciones
        _, predicted_ar = torch.max(val_output.data, 1)
        -, predicted_va = torch.max(ar_output.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = 100 * correct / total
