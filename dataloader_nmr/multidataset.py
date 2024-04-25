import torch
from torch.utils import data
import numpy as np

class MultiDataSet(data.Dataset):
    def __init__(self,
                 inputs: np.ndarray,
                 targets: np.ndarray,
                 transform=None,
                 depth = 5,                 
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.last = inputs.shape[0]-1
        self.zero_plane = np.zeros_like(inputs[0])
        self.depth = depth

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        #input_ID = self.inputs[index]
        #target_ID = self.targets[index]

        # Load input and target
        
        npstack = np.zeros(self.depth*2+1, dtype = np.ndarray)
        for i, ix in enumerate(range (index-self.depth, index+self.depth+1)):
            if (ix < 0 or ix > self.last):
                npstack[i] = self.zero_plane.copy()
            else:
                npstack[i] = self.inputs[ix]
        
        x = np.concatenate(npstack, axis=2) 
        y = self.targets[index]        
        
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)       


        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), \
                  torch.from_numpy(y).type(self.targets_dtype)

        return x, y

class MultiDataSetP(data.Dataset):
    def __init__(self,
                 inputs: np.ndarray,
                 targets: np.ndarray,
                 transform=None,
                 depth = 5,                 
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.last = inputs.shape[0]-1
        self.zero_plane = np.zeros((inputs.shape[1], inputs.shape[2]), dtype=np.float32)
        self.depth = depth

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        #input_ID = self.inputs[index]
        #target_ID = self.targets[index]

        # Load input and target
        
        x = np.zeros((self.inputs.shape[1], self.inputs.shape[2], self.depth*2+1), dtype=np.float32)
        for i, ix in enumerate(range (index-self.depth, index+self.depth+1)):
            if (ix < 0 or ix > self.last):
                x[:,:,i] = self.zero_plane
            else:
                x[:,:,i] = self.inputs[ix,:,:,0]
        

        y = self.targets[index]        
        
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)       


        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), \
                  torch.from_numpy(y).type(self.targets_dtype)

        return x, y
