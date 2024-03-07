import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataset import getImgH5, CustomDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


img, labels = getImgH5()
img = img / 255.0
print(type(img),type(img[0]),type(labels),type(labels[0]))
print(img.shape,labels.shape)
print(img[0].shape,labels[0].shape)

plt.subplot(1, 2, 1)
plt.imshow(img[0])
plt.title('Image')
plt.subplot(1, 2, 2)
plt.imshow(labels[0], cmap='jet')
plt.title('Label')
plt.show()
# x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2)
# x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1)
# x_train, x_test, x_val = x_train / 255.0, x_test / 255.0, x_val / 255.0


# # Create dataset and dataloaders
# train_dataset = CustomDataset(x_train, y_train)
# test_dataset = CustomDataset(x_test, y_test)
# val_dataset = CustomDataset(x_val, y_val)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# n_train, n_val = len(x_train), len(x_val)


# model = UNet(n_channels=3, n_classes=1, bilinear=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device=device)

# epochs = 5
# dir_checkpoint = Path('./checkpoints/')

# criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-8)

# for epoch in range(1, epochs + 1):
#     model.train()
#     epoch_loss = 0
#     with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
#         for inputs, targets in train_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             optimizer.zero_grad()
#             loss.backward()
#             pbar.update(inputs.shape[0])
#             epoch_loss += loss.item()
#             pbar.set_postfix(**{'loss (batch)': loss.item()})
        
#         model.eval()
#         num_val_batches = len(val_loader)
#         mae_score = 0.0
#         for inputs, targets in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)      
#             mae_score += criterion(outputs, targets)     
#         model.train() 
#         val_score = mae_score / num_val_batches

#     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
#     state_dict = model.state_dict()
#     torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
