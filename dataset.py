# File for functions for dataset
import os
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm import tqdm 
import matplotlib.pyplot as plt
import h5py

def trainImgH5(trainortest):
    if trainortest == 'test':
        imagesPath = Path(".\\shanghaitech_data\\ShanghaiTechSplit\\test-data\\images")
        h5Path = Path(".\\shanghaitech_data\\ShanghaiTechSplit\\test-data\\ground-truth-h5")
    elif trainortest == 'train':
        imagesPath = Path(".\\shanghaitech_data\\ShanghaiTechSplit\\train-data\\images")
        h5Path = Path(".\\shanghaitech_data\\ShanghaiTechSplit\\train-data\\ground-truth-h5")
    elif trainortest == 'combined':
        imagesPath = Path(".\\shanghaitech_data\\ShanghaiTechCombined\\images")
        h5Path = Path(".\\shanghaitech_data\\ShanghaiTechCombined\\ground-truth-h5")


    # gtPath = Path(".\\shanghaitech_data\\ShanghaiTechCombinedSplit\\train_data\\ground-truth")

    imagesList = list(imagesPath.glob(r"*.jpg"))
    h5List = list(h5Path.glob(r"*.h5"))
    # gtList = list(gtPath.glob(r"*.mat"))
    # print(len(imagesList),len(gtList))
    imagesList = sorted(imagesList)
    h5List = sorted(h5List)
    # gtList = sorted(gtList)
    # print(len(imagesList),len(gtList))
    # print(imagesList[0],gtList[0])
    # print(imagesList[200],gtList[200])

    h5Series = pd.Series(h5List,name="H5").astype(str)
    imgSeries = pd.Series(imagesList,name="IMAGE").astype(str)
    # matSeries = pd.Series(gtList,name="MAT").astype(str)
    # print(imgSeries[0],matSeries[0])
    # print(imgSeries[200],matSeries[200])

    imgLabelList = []
    h5LabelList = []

    for x_images,x_h5 in tqdm(zip(imgSeries.values,h5Series.values)):
        
        try:
            Reading_Image = cv2.cvtColor(cv2.imread(x_images),cv2.COLOR_BGR2RGB)
            Resized_Image = cv2.resize(Reading_Image,(180,180))
            with h5py.File(x_h5,'r') as f:
                data = f['density'][()]
            Resized_Gaussian_Image = cv2.resize(data,(180,180), interpolation=cv2.INTER_CUBIC)
            
            imgLabelList.append(Resized_Image)
            h5LabelList.append(Resized_Gaussian_Image)
            
        except :
            print('f')
            pass

    X_Train = np.array(imgLabelList,dtype="float32")
    Y_Train = np.array(h5LabelList,dtype="float32")

    return X_Train, Y_Train


