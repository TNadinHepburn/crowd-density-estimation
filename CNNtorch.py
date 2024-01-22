import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from dataset import trainImgH5

# Define the Convolutional Neural Network
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=2, padding=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
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


img, labels = trainImgH5(trainortest='combined')
x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1)
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train[0].shape,y_train[0].shape)

# Create dataset and dataloaders
train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

img0, dm = train_dataset[0]
figure,axis = plt.subplots(1,2,figsize=(10,10))
axis[0].imshow(np.transpose(img0,(1,2,0)), cmap="jet")
axis[0].set_xlabel(img0.shape)
axis[0].set_title("MAT")
axis[1].imshow(dm)
axis[1].set_xlabel(dm.shape)
axis[1].set_title("ORIGINAL")
plt.show()

# Create an instance of the model
model = SimpleConvNet()

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.nn.functional.interpolate(outputs, size=(180, 180), mode='bilinear', align_corners=False)
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
            outputs = torch.nn.functional.interpolate(outputs, size=(180, 180), mode='bilinear', align_corners=False)
            total_loss += criterion(outputs, targets).item()

        average_loss = total_loss / len(test_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {average_loss}')

# Testing
model.eval()
print('x_test',x_test[0].shape)
sample_input = torch.tensor(np.transpose(x_test[0],(2,0,1)), dtype=torch.float32).to(device)
output_heatmap = model(sample_input)
output_heatmap_np = output_heatmap.cpu().detach().numpy().squeeze()

# Visualize the output heatmap
plt.imshow(output_heatmap_np, cmap='coolwarm')
plt.title("Density Heatmap (Test Sample)")
plt.show()

figure,axis = plt.subplots(1,3,figsize=(10,10))
axis[0].imshow(x_test[0], cmap="jet")
axis[0].set_xlabel(x_test[0].shape)
axis[0].set_title("MAT")
axis[1].imshow(output_heatmap_np)
axis[1].set_xlabel(output_heatmap_np.shape)
axis[1].set_title("ORIGINAL")
axis[2].imshow(y_test[0])
axis[2].set_xlabel(y_test[0].shape)
axis[2].set_title("GT")
plt.show()
