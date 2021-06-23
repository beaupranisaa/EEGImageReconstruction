import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu = f'cuda:{np.argmax(memory_available)}'
    return gpu

def get_electrode(X, electrodes):
    XX = []
    for electrode in electrodes:
        x = X[:, electrode,:]
        XX.append(x)
    XX = np.array(XX)
    return np.transpose(XX,(1,0,2))

def check_split(X1, X2, y1, y2, name1, name2):
    unique1, count1 = np.unique(y1, return_counts=True)
    unique2, count2 = np.unique(y2, return_counts=True)
    print(count1[0],count1[1],count1[2])
    assert count1[0] == count1[1] == count1[2]
    assert count2[0] == count2[1] == count2[2]

    print('='*20,name1,'='*20)
    print(f"Shape of X_{name1}: ", X1.shape)
    print(f"Shape of y_{name1}: ",y1.shape)
    print(f"Classes of y_{name1}: ",unique1)
    print(f"Counts of y_{name1} classes: ",count1)
    print('='*20,name2,'='*20)
    print(f"Shape of X_{name2}: ",X2.shape)
    print(f"Shape of y_{name2}: ",y2.shape)
    print(f"Classes of y_{name2}: ",unique2)
    print(f"Counts of y_{name2} classes: ",count2)
    
def chunk_data(data, size):
    data_keep = data.shape[2] - (data.shape[2]%size)
    data = data[:,:,:data_keep]
    data = data.reshape(-1,data.shape[1],data.shape[2]//size,size)
    data = np.transpose(data, (0, 2, 1, 3)  )
    return data

def filled_y(y, chunk_num):
    yy = np.array([[i] *chunk_num for i in  y ]).ravel()
    return yy

def check_torch_shape(torch_X, torch_y, name):
    print('='*20,name,'='*20)
    print(f"Shape of torch_X_{name}: ",torch_X.shape)
    print(f"Shape of torch_y_{name}: ",torch_y.shape)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def do_plot(train_losses, valid_losses):
    plt.figure(figsize=(25,5))
#     clear_output(wait=True)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title('Train and Val loss')
    plt.legend()
    plt.show()

#Count the parameters for writing papers
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def squeeze_to_list(_tmp):
    from functools import reduce
    import operator
    xx = [ i.cpu().detach().numpy().ravel().tolist() for i in _tmp]
    xx = reduce(operator.concat, xx)
    return xx
