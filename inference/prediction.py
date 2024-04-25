import torch

import numpy as np
from tqdm import tqdm

import torch
import numpy as np
#from skimage import morphology

def predict_batch(
    model,
    test_loader,
    device,
    notebook=False,
    dtype = "uint8",
    regression = False
) -> np.ndarray:
    """
    Takes in images and outputs final preditions
        :param model: a PyTorch Model object
        :param test_loader: torch dataloader
        :param notebook: bool to define the representation of progress bar        
        :return: predicted full-sized images
        
    """
 
    #num_channels = img_data.shape[-1] 

    try:
        tqdm._instances.clear()  # type: ignore
    except:
        pass

    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange
    
    final_full_preds = []
    
    model.eval()
    batch_iter = tqdm(enumerate(test_loader), 'Prediction', total=len(test_loader),
                     leave=False)
    
    for i, (x, y) in batch_iter:
            
        x = x.to(device)
        with torch.no_grad():
            out_test = model(x)    

        if regression:
            out_test = out_test.cpu().numpy()
        else:    
            out_test = torch.softmax(out_test, dim=1)
            out_test = torch.argmax(out_test, dim=1).cpu().numpy()        
        final_full_preds.append(out_test)
    final_full_preds = np.concatenate(final_full_preds)

    final_full_preds = final_full_preds.astype(dtype = dtype)
    return final_full_preds
    
    
def predict_batch_cpu(
    model,
    test_loader,
    device,
    notebook=False,
    dtype = np.uint8
) -> np.ndarray:
    """
    Takes in images and outputs final preditions
        :param model: a PyTorch Model object
        :param test_loader: torch dataloader
        :param notebook: bool to define the representation of progress bar        
        :return: predicted full-sized images
        
    """
 
    #num_channels = img_data.shape[-1] 

    try:
        tqdm._instances.clear()  # type: ignore
    except:
        pass

    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange
    
    final_full_preds = []
    
    model.eval()
    batch_iter = tqdm(enumerate(test_loader), 'Prediction', total=len(test_loader),
                     leave=False)
    
    for i, (x, y) in batch_iter:
            
        #x = x.to(device)
        with torch.no_grad():
            out_test = model(x)    

        out_test = torch.softmax(out_test, dim=1)
        out_test = torch.argmax(out_test, dim=1).cpu().numpy()        
        final_full_preds.append(out_test)
    final_full_preds = np.concatenate(final_full_preds)

    final_full_preds = final_full_preds.astype(dtype = dtype)
    return final_full_preds    

