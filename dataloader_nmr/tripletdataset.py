import torch
from torch.utils import data
import numpy as np

class TripletDataSet(data.Dataset):
    def __init__(self,
                 inputs: np.ndarray,
                 targets: np.ndarray,
                 transform=None                                 
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.last = inputs.shape[0]-1

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        #input_ID = self.inputs[index]
        #target_ID = self.targets[index]

        # Load input and target
        ix_pre = index if index==0 else index-1
        ix_post = index if index==self.last else index+1
        x_pre, y_pre = self.inputs[ix_pre], self.targets[ix_pre]
        x, y = self.inputs[index], self.targets[index]
        x_post, y_post = self.inputs[ix_post], self.targets[ix_post]
        x = np.concatenate((x_pre, x, x_post), axis=2)

        
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)       


        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), \
                  torch.from_numpy(y).type(self.targets_dtype)

        return x, y