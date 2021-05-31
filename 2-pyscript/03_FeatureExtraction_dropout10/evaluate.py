import torch
import numpy as np

def evaluate(model, iterator, criterion, classes,device, test = False):
    
    total = 0
    correct = 0
    epoch_loss = 0
    epoch_acc = 0
    predicted_list = []
    labels_list    = []
    
    class_correct = np.zeros(len(classes))
    class_total = np.zeros(len(classes))
    
    model.eval()
    
    with torch.no_grad():
    
        for batch, labels in iterator:
            
            #Move tensors to the configured device
            batch = batch.to(device)
            labels = labels.to(device)

            predictions = model(batch.float())
            loss = criterion(predictions, labels.long())
        
            _, predicted = torch.max(predictions.data, 1)  #returns max value, indices
            #print(predicted)
#             clear_output(wait=True)
#             print('================== Predicted y ====================')
#             print(predicted) 
#             print('==================    True y   ====================')
#             print(labels)  
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
            total += labels.size(0)  #keep track of total
            correct += (predicted == labels).sum().item()  #.item() give the raw number
            acc = 100 * (correct / total)
            
            epoch_loss += loss.item()
            epoch_acc += acc
            
            labels_list.append(labels)
            predicted_list.append(predicted)
    
    if test == True:
        acc_class = {classes[cl]: [] for cl in range(len(classes))}
        
        for i in range(len(classes)):
            acc_class[classes[i]].append(100 * np.float64(class_correct[i])/ class_total[i])
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * np.float64(class_correct[i])/ class_total[i]))   
    if test == True:
        return epoch_loss / len(iterator), epoch_acc / len(iterator) ,predicted_list, labels_list, acc_class
    else:
        return epoch_loss / len(iterator), epoch_acc / len(iterator) ,predicted_list, labels_list

