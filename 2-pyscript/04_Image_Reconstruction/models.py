import torch
import torch.nn as nn
import torch.nn.functional as F

import fnmatch

class EEGEncoder(nn.Module):
    '''
    Expected Input Shape: (batch, channels, height , width)
    '''
    def __init__(self):
        super().__init__()
        
        self.activation = nn.Tanh()
        
        self.conv1 = nn.Sequential(    nn.Conv1d(16, 32, kernel_size=(1,3),   padding=(0,0), stride=(1,1))  ,  self.activation )
        self.conv2 = nn.Sequential(    nn.Conv1d(32, 64, kernel_size=(1,3) ,  padding=(0,0), stride=(1,1))  ,  self.activation )
        # nn.Linear(XY,256) need to be changed!
        self.fc1   = nn.Sequential(    nn.Linear(384,256),  self.activation ,nn.Dropout(0.1)   ,nn.BatchNorm1d(256)   )
        self.fc2   = nn.Sequential(    nn.Linear(256,128),  self.activation ,nn.Dropout(0.1)   ,nn.BatchNorm1d(128) )
        self.fc3   = nn.Sequential(    nn.Linear(128,64),  self.activation  ,nn.Dropout(0.1)   ,nn.BatchNorm1d(64) )
        self.fc4   = nn.Sequential(    nn.Linear(64,32),  self.activation   ,nn.Dropout(0.1)   ,nn.BatchNorm1d(32) )
        self.fc5   = nn.Sequential(    nn.Linear(32,3)   )

        self.is_debug= False
        
    def encode(self, X):
        
        
        if self.is_debug  : print('--------Convolute--------'); print(X.shape) 
            
        X = self.conv1(X)
        if self.is_debug  : print(X.shape) 
            
        X = self.conv2(X)
        if self.is_debug  : print(X.shape) 
            
        X = X.flatten(start_dim = 1)

        # print(X.shape) 
 
        X = self.fc1(X)
        if self.is_debug : print('--------Flatten--------') ; print(X.shape) 

        X = self.fc2(X)
        if self.is_debug  : print(X.shape) 

        X = self.fc3(X)
        if self.is_debug  : print(X.shape) 
        
        X = self.fc4(X)
        if self.is_debug  : print(X.shape) 

        X = self.fc5(X)
        if self.is_debug  : print(X.shape) 

            
        return X
        
    def forward(self,X):
        X = self.encode(X)
        return X
    
    def get_latent( self, X):
        if self.is_debug  : print('--------Convolute--------'); print(X.shape) 
            
        X = self.conv1(X)
        if self.is_debug  : print(X.shape) 
            
        X = self.conv2(X)
        if self.is_debug  : print(X.shape) 
            
        X = X.flatten(start_dim = 1)
        if self.is_debug  : print('--------Flatten--------') ; print(X.shape) 
 
        X = self.fc1(X)
        if self.is_debug : print('--------Flatten--------') ; print(X.shape) 

        X = self.fc2(X)
        if self.is_debug  : print(X.shape) 

        X = self.fc3(X)
        if self.is_debug  : print(X.shape) 
        
        X = self.fc4(X)
        if self.is_debug  : print(X.shape) 
        
        return X
    
    def classifier(self, latent):
        return self.fc5(latent)


class Generator(nn.Module):
    """
    Input : random noise / latent vector of any size
    Output : Fake images same size as Real images 
    torch.Size([3, 48])
    torch.Size([3, 96])
    torch.Size([3, 192])
    torch.Size([3, 384])
    torch.Size([3, 576])
    torch.Size([3, 768])
    torch.Size([3, 150528])
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        
        self.activation = nn.LeakyReLU()
        
        self.fc1 = nn.Sequential (nn.Linear(input_size    , hidden_size )  , self.activation, nn.Dropout(0.3) )
        self.fc2 = nn.Sequential (nn.Linear(hidden_size   , hidden_size*2) , self.activation, nn.Dropout(0.3) ) 
        self.fc3 = nn.Sequential (nn.Linear(hidden_size*2 , hidden_size*4) , self.activation, nn.Dropout(0.3) ) 
        self.fc4 = nn.Sequential (nn.Linear(hidden_size*4 , hidden_size*6) , self.activation, nn.Dropout(0.3) ) 
        self.fc5 = nn.Sequential (nn.Linear(hidden_size*6 , hidden_size*8) , self.activation, nn.Dropout(0.3) )
        self.fc6 = nn.Sequential (nn.Linear(hidden_size*8 , output_size )  , self.activation, )
        
    def forward(self, X, noise):
        X = torch.cat([X, noise], dim = 1)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        X = self.fc5(X)
        X = self.fc6(X)
        X = X.reshape(-1, 3, 224, 224)
        return X

class Discriminator(nn.Module):
    """
    Input : Real / Fake images
    Output : Classification Real = 1 / Fake = 0
    """
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        
        self.activation = nn.LeakyReLU()
        
        self.fc1 = nn.Sequential (nn.Linear(input_size, hidden_size*8)      , self.activation, nn.Dropout(0.3))
        self.fc2 = nn.Sequential (nn.Linear(hidden_size*8, hidden_size*6)   , self.activation, nn.Dropout(0.3))
        self.fc3 = nn.Sequential (nn.Linear(hidden_size*6, hidden_size*4)   , self.activation, nn.Dropout(0.3))
        self.fc4 = nn.Sequential (nn.Linear(hidden_size*4, hidden_size*2)   , self.activation, nn.Dropout(0.3))
        self.fc5 = nn.Sequential (nn.Linear(hidden_size*2, hidden_size)   , self.activation  , nn.Dropout(0.3))
        self.fc6 = nn.Sequential (nn.Linear((hidden_size)+32, hidden_size), self.activation  , nn.Dropout(0.3))
        
        self.fc5_0 = nn.Sequential (nn.Linear(hidden_size, 32), self.activation  , nn.Dropout(0.3) )  # for real / fake
        self.fc5_1 = nn.Sequential (nn.Linear(32, 16), self.activation  , nn.Dropout(0.3) )
        self.fc5_2 = nn.Sequential (nn.Linear(16, 8),self.activation  , nn.Dropout(0.3)  ) 
        self.fc5_3 = nn.Sequential (nn.Linear(8, 1)  )
        
        self.fc6_0 = nn.Sequential (nn.Linear(hidden_size, 32) , self.activation  , nn.Dropout(0.3)  ) # for number classification
        self.fc6_1 = nn.Sequential (nn.Linear(32, 3) )
    
    def forward(self, X, latent):
#         print('DISCRIMINATOR')
               
        X = X.flatten(start_dim = 1)

        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        X = self.fc5(X)
        X = torch.cat([X, latent], dim = 1)                          
        X = self.fc6(X)
        
        rf_decision = self.fc5_0(X)
        rf_decision = self.fc5_1(rf_decision)
        rf_decision = self.fc5_2(rf_decision)
        rf_decision = self.fc5_3(rf_decision)
        
        num_decision = self.fc6_0(X)
        num_decision = self.fc6_1(num_decision)
        
        return rf_decision, num_decision