import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import getImgH5

# torch.cuda.empty_cache()
# print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Define the Convolutional Neural Network
class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=5, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=5, padding=1)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x
    
class ComplexConvNetwork(nn.Module):
    def __init__(self):
        super(ComplexConvNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Adjusted for input size 224x224
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = torch.tensor(np.transpose(self.x_data[index],(2,0,1)), dtype=torch.float32)
        y = torch.tensor(self.y_data[index], dtype=torch.float32)
        return x, y


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

# Create an instance of the model
model = ConvNetwork()
print('creating conv network',model)

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.nn.functional.interpolate(outputs, size=(224, 224), mode='bilinear')
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)            
            outputs = torch.nn.functional.interpolate(outputs, size=(224, 224), mode='bilinear')
            total_loss += criterion(outputs, targets).item()

        average_loss = total_loss / len(test_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {average_loss}')

# Testing
counter = 0
model.eval()
for img, gt in zip(x_train[:5],y_train[:5]):
    sample_input = torch.tensor(np.transpose(img,(2,0,1)), dtype=torch.float32).to(device)
    output_heatmap = model(sample_input)
    output_heatmap_np = output_heatmap.cpu().detach().numpy().squeeze()
    output_heatmap_np = output_heatmap_np / 255.0

    # Visualize the output heatmap
    # plt.imshow(output_heatmap_np, cmap='jet')
    # plt.title("Density Heatmap (Test Sample)")
    # plt.show()
    
    print('np:', output_heatmap_np)
    print('gt:', gt)

    figure,axis = plt.subplots(1,3,figsize=(10,10))
    axis[0].imshow(img, cmap="jet")
    axis[0].set_xlabel(img.shape)
    axis[0].set_title("ORIGINAL")
    axis[1].imshow(output_heatmap_np, cmap='jet')
    axis[1].set_xlabel(output_heatmap_np.shape)
    axis[1].set_title("PREDICTION")
    axis[2].imshow(gt)
    axis[2].set_xlabel(gt.shape)
    axis[2].set_title("ACTUAL")
    plt.savefig(f'SLIGHTCONV2D_TEST_{counter}.png')
    plt.show()
    counter += 1
