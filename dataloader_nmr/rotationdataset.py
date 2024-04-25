import torch
from torch.utils import data
import numpy as np
from  pandas import DataFrame

class RotationDataSet(data.Dataset):
    def __init__(self,
                 inputs: np.ndarray,
                 dataframe: DataFrame,
                 transform=None,
                 sine = False                                 
                 ):
        self.inputs = inputs
        self.dataframe = dataframe
        self.transform = transform 
        self.sine = sine       
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):

        # Load input and target  & preprocessing
        if self.sine:
            x, y = self.inputs[index], self.dataframe.loc[index, ["sin", "cos"]]
            y = y.values.astype(np.float32)
        else:
            x, y = self.inputs[index], self.dataframe.loc[index, 'rad']
            y = np.expand_dims(np.array(y/np.pi), 0)  # type: ignore       
        
        
        if self.transform is not None:
            x, y = self.transform(x, y)
        

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), \
               torch.from_numpy(y).type(self.targets_dtype)

        return x, y