# File for functions evaluation
from tqdm import tqdm
import torch
import torch.nn as nn

def evaluate(model, val_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = nn.L1Loss()
    num_val_batches = len(val_data)
    mse_score = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(val_data, total=num_val_batches, desc='Evaluation round', unit='batch', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            mse_score += criterion(predictions, targets)
        model.train()
    return (mse_score / max(num_val_batches, 1)).item()

