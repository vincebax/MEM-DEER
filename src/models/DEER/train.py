import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(model, dataloader, epochs):
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.001)

    num_batches = len(dataloader)

    for epoch in range(epochs):

        epoch_loss = 0

        for image, scanpath, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(image, scanpath)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f'epoch {epoch} loss: {epoch_loss / num_batches}')
        
