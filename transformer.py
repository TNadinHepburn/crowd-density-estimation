# File of functions for transformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from CNNtorch import CustomDataset
from dataset import getImgH5
from sklearn.model_selection import train_test_split


# Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, input_channels=3, patch_size=16, num_patches=32, hidden_size=256, num_heads=4, num_layers=6):
        super(VisionTransformer, self).__init__()

        self.patch_embedding = nn.Conv2d(input_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).permute(2, 0, 1)  # (num_patches, batch_size, hidden_size)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global average pooling
        x = self.fc(x)
        return x


img, labels = getImgH5()
x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1)
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train[0].shape,y_train[0].shape)


# Create dataset and dataloaders
train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)
val_dataset = CustomDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



model = VisionTransformer()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train Model
num_epochs = 40
for epoch in range(num_epochs):
    model.train()    
    for inputs, targets in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


    # # Validation
    # model.eval()
    # with torch.no_grad():
    #     total_loss = 0.0
    #     for inputs, targets in test_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)            
    #         outputs = torch.nn.functional.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
    #         total_loss += criterion(outputs, targets).item()

    #     average_loss = total_loss / len(test_loader)
    #     print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {average_loss}')


# Testing
print('testing')
counter = 0
model.eval()
for img, gt in zip(x_test[:5],y_test[:5]):
    sample_input = torch.tensor(np.transpose(img,(2,0,1)), dtype=torch.float32).to(device)
    output_heatmap = model(sample_input)
    output_heatmap_np = output_heatmap.cpu().detach().numpy().squeeze()

    # Visualize the output heatmap
    # plt.imshow(output_heatmap_np, cmap='jet')
    # plt.title("Density Heatmap (Test Sample)")
    # plt.show()
    figure,axis = plt.subplots(1,3,figsize=(10,10))
    axis[0].imshow(img, cmap="jet")
    axis[0].set_xlabel(img.shape)
    axis[0].set_title("ORIGINAL")
    axis[1].imshow(abs(output_heatmap_np), cmap='jet')
    axis[1].set_xlabel(output_heatmap_np.shape)
    axis[1].set_title("PREDICTION")
    axis[2].imshow(gt)
    axis[2].set_xlabel(gt.shape)
    axis[2].set_title("ACTUAL")
    plt.savefig(f'SIMPLETRANS_TEST_{counter}.png')
    plt.show()
    counter += 1

print('fin')