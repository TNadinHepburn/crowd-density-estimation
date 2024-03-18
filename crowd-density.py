# Main file for crowd density estimation
from evaluation import evaluate
from CNNunet import UNet
from CNNunettrain import train_model
import torch
from sklearn.model_selection import train_test_split
from dataset import getImgH5, CustomDataset
from torch.utils.data import DataLoader
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--evaluation', '-v', metavar='EVAL', type=bool, default=False, help='Evaulate on loaded model')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    img, labels = getImgH5()
    x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, random_state=1)

    # Create dataset and dataloaders
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    val_dataset = CustomDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3,n_classes=1,bilinear=True)

    if args.load:
        model.to(device=device)
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)

    if args.evaluation:
        mae_result = evaluate(model, val_data=val_loader)
        print(mae_result)

    elif not args.evaluation:
        train_model(model=model, epochs=10, train_dataset=train_loader, val_dataset=val_loader, device=device, n_train=len(x_train))