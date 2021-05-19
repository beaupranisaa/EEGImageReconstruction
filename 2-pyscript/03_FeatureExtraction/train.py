import torch
import numpy as np

def train(model, iterator, optimizer, criterion, device):
    total = 0
    correct = 0
    epoch_loss = 0
    epoch_acc = 0
    predicted_list = []
    model.train()
    
    for batch, labels in iterator:
        
        #Move tensors to the configured device
        batch  = batch.to(device)
        labels = labels.to(device)
       
        
        #Forward pass
        outputs = model(batch.float())
        outputs = outputs.to(device)
        
        loss = criterion(outputs, labels.long()).to(device)

        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        #check accuracy
        predictions = model(batch.float())
        _, predicted = torch.max(predictions.data, 1)  #returns max value, indices
        total += labels.size(0)  #keep track of total
        correct += (predicted == labels).sum().item()  #.item() give the raw number
        acc = 100 * (correct / total)
                
        epoch_loss += loss.item()
        epoch_acc = acc
        predicted_list.append(predicted)
        
    return epoch_loss / len(iterator), epoch_acc, predicted_list

