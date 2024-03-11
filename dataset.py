# File for functions for dataset
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm 
import h5py
from torch.utils.data import Dataset
import torch

def getImgH5():
    imagesPath = Path(".\\shanghaitech_data\\ShanghaiTechCombined\\images")
    h5Path = Path(".\\shanghaitech_data\\ShanghaiTechCombined\\ground-truth-h5")

    imagesList = list(imagesPath.glob(r"*.jpg"))
    h5List = list(h5Path.glob(r"*.h5"))
    imagesList = sorted(imagesList)
    h5List = sorted(h5List)

    h5Series = pd.Series(h5List,name="H5").astype(str)
    imgSeries = pd.Series(imagesList,name="IMAGE").astype(str)

    imgLabelList = []
    h5LabelList = []

    for x_images,x_h5 in tqdm(zip(imgSeries.values,h5Series.values)):
        
        try:
            Reading_Image = cv2.cvtColor(cv2.imread(x_images),cv2.COLOR_BGR2RGB)
            Resized_Image = cv2.resize(Reading_Image,(224,224))
            with h5py.File(x_h5,'r') as f:
                data = f['density'][()]
            Resized_Gaussian_Image = cv2.resize(data,(224,224))
            
            imgLabelList.append(Resized_Image)
            h5LabelList.append(Resized_Gaussian_Image)
            
        except :
            print('f')
            pass

    X_Train = np.array(imgLabelList,dtype="float32")
    Y_Train = np.array(h5LabelList,dtype="float32")
    Y_Train = np.expand_dims(Y_Train, axis=-1)

    return X_Train, Y_Train

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = torch.tensor(np.transpose(self.x_data[index],(2,0,1)), dtype=torch.float32)
        y = torch.tensor(np.transpose(self.y_data[index],(2,0,1)), dtype=torch.float32)
        return x, y
