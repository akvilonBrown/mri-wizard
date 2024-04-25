import torch
from torch.utils import data
import numpy as np

class TwoDimDataSet(data.Dataset):
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

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        #input_ID = self.inputs[index]
        #target_ID = self.targets[index]

        # Load input and target
        x, y = self.inputs[index], self.targets[index]

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)       


        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), \
                  torch.from_numpy(y).type(self.targets_dtype)

        return x, y