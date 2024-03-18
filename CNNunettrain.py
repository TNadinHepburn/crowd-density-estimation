import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from pathlib import Path
import datetime

def train_model(model, epochs, train_dataset, val_dataset, device, n_train):

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-8)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for inputs, targets in train_dataset:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                pbar.update(inputs.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            
            model.eval()
            num_val_batches = len(val_dataset)
            mae_score = 0.0
            with torch.no_grad():
                for inputs, targets in tqdm(val_dataset, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)      
                    mae_score += criterion(outputs, targets)     
                model.train() 
                val_score = mae_score / num_val_batches

        dir_checkpoint = Path('./checkpoints/')

        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, str(dir_checkpoint / '{}checkpoint_epoch{}.pth'.format(datetime.datetime.now().strftime("%m%d%H%M"),epoch)))

    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    new_model_path = '{}FINAL_EPOCH.pth'.format(datetime.datetime.now().strftime("%m%d%H%M"))
    torch.save(state_dict, str(dir_checkpoint / new_model_path))
    print('Model saved at {}'.format(new_model_path))