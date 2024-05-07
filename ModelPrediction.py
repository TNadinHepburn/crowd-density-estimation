from sklearn.model_selection import train_test_split
from UNet import UNet
from dataset import getImgH5, CustomDataset
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime

def predict_img(net,
                x_test,
                device,
                scale_factor=1):
    net.eval()
    img = x_test
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (x_test.size[1], x_test.size[0]), mode='bilinear')
        
    return output[0].long().squeeze().numpy()

if __name__ == '__main__':
    img, labels = getImgH5()
    img = img / 255.0

    x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, random_state=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model.to(device=device)
    state_dict = torch.load('checkpoints/checkpoint_epoch5.pth', map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    counter = 0
    for img, gt in zip(x_train[:5],y_train[:5]):
        sample_input = torch.tensor(np.transpose(img,(2,0,1)), dtype=torch.float32).to(device)
        output_heatmap = model(sample_input)
        output_heatmap_np = output_heatmap.cpu().detach().numpy().squeeze()
        output_heatmap_np = output_heatmap_np / 255.0

        figure,axis = plt.subplots(1,3,figsize=(10,10))
        axis[0].imshow(img)
        axis[0].set_xlabel(img.shape)
        axis[0].set_title("ORIGINAL")
        axis[1].imshow(output_heatmap_np, cmap='jet')
        axis[1].set_xlabel(output_heatmap_np.shape)
        axis[1].set_title("PREDICTION")
        axis[2].imshow(gt, cmap="jet")
        axis[2].set_xlabel(gt.shape)
        axis[2].set_title("ACTUAL")
        plt.savefig(f'Results/{datetime.datetime.now().strftime("%m%d%H%M")}_TEST_{counter}.png')
        plt.show()
        counter += 1