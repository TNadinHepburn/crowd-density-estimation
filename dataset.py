# File for functions for dataset
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm 
import h5py

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
            Resized_Gaussian_Image = cv2.resize(data,(224,224), interpolation=cv2.INTER_CUBIC)
            
            imgLabelList.append(Resized_Image)
            h5LabelList.append(Resized_Gaussian_Image)
            
        except :
            print('f')
            pass

    X_Train = np.array(imgLabelList,dtype="float32")
    Y_Train = np.array(h5LabelList,dtype="float32")

    return X_Train, Y_Train
