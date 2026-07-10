import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy

def train_epoch(model, dataloader, device, criterion, optimizer):
    model.train()

    num_samples = 0
    running_loss = 0
    running_accuracy = 0

    for image, scanpath, labels in dataloader:
        
        image = image.to(device)
        scanpath = scanpath.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(image, scanpath)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        num_samples += batch_size
        running_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        running_accuracy += (preds == labels).sum().item()
    
    epoch_loss = running_loss / num_samples
    epoch_accuracy = running_accuracy / num_samples

    return epoch_loss, epoch_accuracy

def evaluate(model, dataloader, device, criterion):
    model.eval()

    num_samples = 0
    running_loss = 0
    running_accuracy = 0

    with torch.no_grad():
        for image, scanpath, labels in dataloader:
            
            image = image.to(device)
            scanpath = scanpath.to(device)
            labels = labels.to(device)

            outputs = model(image, scanpath)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            num_samples += batch_size
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            running_accuracy += (preds == labels).sum().item()
            
    epoch_loss = running_loss / num_samples
    epoch_accuracy = running_accuracy / num_samples
    
    return epoch_loss, epoch_accuracy

def save_checkpoint(model_dir, model, optimizer, epoch, loss):
    model_checkpoint = {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'epoch' : epoch,
        'loss' : loss
    }

    torch.save(model_checkpoint, model_dir / 'best_checkpoint.pt')


def train(model_dir, model, training_dataloader, validation_dataloader, device, criterion, optimizer, epochs=5):

    metrics = {
        'training_losses' : [],
        'training_accuracies' : [],
        'validation_losses' : [],
        'validation_accuracies' : []
    }

    best_validation_loss = float('inf')
    
    for epoch in range(epochs):

        train_epoch_loss, train_epoch_accuracy = train_epoch(model, training_dataloader, device, criterion, optimizer)
        metrics['training_losses'].append(train_epoch_loss)
        metrics['training_accuracies'].append(train_epoch_accuracy)

        validation_epoch_loss, validation_epoch_accuracy = evaluate(model, validation_dataloader, device, criterion)
        metrics['validation_losses'].append(validation_epoch_loss)
        metrics['validation_accuracies'].append(validation_epoch_accuracy)

        if validation_epoch_loss < best_validation_loss:
            best_validation_loss = validation_epoch_loss
            save_checkpoint(model_dir, model, optimizer, epoch + 1, best_validation_loss)
        
        print(f'epoch {epoch + 1} | train_loss: {train_epoch_loss:.4f}, train_accuracy: {train_epoch_accuracy:.4f} | val_loss: {validation_epoch_loss:.4f}, val_accuracy: {validation_epoch_accuracy:.4f}')
    
    return metrics
        