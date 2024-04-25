import numpy as np
import torch
import os
import logging
import torch.optim
from torch.utils.data import DataLoader

# A logger for this file
log = logging.getLogger(__name__)

# Learning rate is updated after each step (not epoch)
class Trainer_lr:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: DataLoader,
                 validation_DataLoader: DataLoader,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 best_model_dir: str = "",
                 saving_milestone: int = 5
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        
        if(best_model_dir):
            #assert os.path.exists(best_model_dir),  "Path for saving best model invalid: " +  best_model_dir
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            self.best_model_dir = best_model_dir
            if not saving_milestone:
                saving_milestone = epochs // 2  # start saving the best from the halfway of training
            self.saving_milestone =  saving_milestone
            #print(self.saving_milestone)
 
    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
                #print(f"epoch: {self.epoch}  val_loss size: {len(self.validation_loss)}") 
                #print(f"val loss: {self.validation_loss[self.epoch-1]}, minL {min(self.validation_loss)}") 
                if(self.best_model_dir and 
                   self.epoch >= self.saving_milestone and 
                   self.validation_loss[i] <= min(self.validation_loss)):                               
                    self._save_model()
     
                
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            inp, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(inp)  # one forward pass
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            self.lr_scheduler.step() 
            self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        #self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            inp, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(inp)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()
    
    def _save_model(self):
        model_name = "best_model_" + str(self.epoch) + ".pt"
        model_checkpoint = "best_model.pt"
        path = os.path.join(self.best_model_dir, model_name)
        path_checkpoint = os.path.join(self.best_model_dir, model_checkpoint)
        try:
            #torch.save(self.model.state_dict(), path)
            #log.info("Saving better checkpoint, epoch: " + str(self.epoch))  # don't use it, it breaks progress bar
            with open(path, mode='a'): pass  #saving a dummy file with epoch name, sort of logging
            torch.save(self.model.state_dict(), path_checkpoint)
        except:
            print("Something went wrong during the model saving")  