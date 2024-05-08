# Main file for crowd density estimation
from ModelEvaluation import evaluate
from UNet import UNet
from ModelTraining import train_model
from ModelPrediction import predict_img
from TransUNet import TransUNet
import torch
from sklearn.model_selection import train_test_split
from dataset import getImgH5, CustomDataset
from torch.utils.data import DataLoader
import argparse
from visulisation import inputsOutputsGT

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--modeltype', '-t', metavar='T', type=str, default='CNN', help='Type of model to use. (CNN, TRANS)')
    parser.add_argument('--evaluation', '-v', metavar='EVAL', type=bool, default=False, help='Evaulate on loaded model')
    parser.add_argument('--predict', '-p', metavar='PRED', type=bool, default=False, help='Predict on test images')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    img, labels = getImgH5()
    x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.3, random_state=1)
    x_test, x_val, y_test, y_val = train_test_split(x_test,y_test, test_size=(10/30), random_state=1)

    # Create dataset and dataloaders
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    val_dataset = CustomDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.modeltype == 'CNN':
        print('Initiating CNN Model')
        model = UNet(n_channels=3,n_classes=1,bilinear=True)
    elif args.modeltype == 'TRANS':
        print("Initiating Transformer Model")
        model = TransUNet(img_dim=224,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)

    if args.load:
        print('Loading Model...')
        model.to(device=device)
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        print('Loading Complete')


    if args.evaluation:
        print('Evaluating Model...')
        result = evaluate(model, val_data=val_loader)
        print('Evaluation Complete')
        print(result)

    if args.predict:
        print('Predicting Model on Test Data...')
        model.to(device=device)
        for img,gt in train_dataset:
            result = predict_img(model, img ,device)
            inputsOutputsGT(img,gt,result)

    if (not args.evaluation) and (not args.predict):
        print('Begining Model Training...')
        model.to(device=device)
        results = train_model(model=model, epochs=args.epochs, train_dataset=train_loader, val_dataset=val_loader, device=device, n_train=len(x_train))
        print('Training Complete')
        print(results)