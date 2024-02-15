from kaggle.api.kaggle_api_extended import KaggleApi
import random
from tqdm import tqdm
import os, shutil

def DownloadKaggleDataset():
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    dataset_name = 'tthien/shanghaitech-with-people-density-map'
    api.dataset_download_files(dataset_name, path='./shanghaitech_data', unzip=True)
    # setup code from https://www.kaggle.com/code/donkeys/kaggle-python-api

def CombineDatasetParts():
    # Combine Part A & B
    combinedDir = './shanghaitech_data/ShanghaiTechCombined/'

    if not os.path.exists(combinedDir):
        os.makedirs(combinedDir)
        for folder in ['ground-truth','ground-truth-h5','images']:
            counter = 0
            for split in ['train_data','test_data']:
            # Iterate through images, gt, and gt-m5 folders
                source_a_path = os.path.join('./shanghaitech_data/ShanghaiTech/part_A/', split, folder)  # Modified this line
                source_b_path = os.path.join('./shanghaitech_data/ShanghaiTech/part_B/', split, folder)  # Modified this line
                destination_path = os.path.join(combinedDir, folder)  # Modified this line

                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                for partPath in [source_a_path, source_b_path]:
                    for filename in tqdm(os.listdir(partPath)):
                        source_file_path = os.path.join(partPath, filename)
                        destination_file_path = os.path.join(destination_path, f'{counter}.{filename.split(".")[-1]}')  # Modified this line
                        counter += 1
                        shutil.copy2(source_file_path, destination_file_path)

def ResplitDataset():
    combinedPath = '.\\shanghaitech_data\\ShanghaiTechCombined'
    trainPath = 'train-data'
    testPath = 'test-data'
    source_path = os.path.join(combinedPath, 'images')
    files = os.listdir(source_path)
    num_files = len(files)
    num_train_files = int(num_files * 0.8)
    file_range = range(0,num_files)
    train_files_index = random.sample(file_range, num_train_files)
    test_files_index = list(set(file_range)-set(train_files_index))

    
    destination_path = '.\\shanghaitech_data\\ShanghaiTechSplit'
    image_path = 'images'
    mat_path = 'ground-truth'
    h5_path = 'ground-truth-h5'

    for fileInts in train_files_index:
        source_file_image = os.path.join(combinedPath, image_path,f'{fileInts}.jpg')
        source_file_mat = os.path.join(combinedPath, mat_path,f'{fileInts}.mat')
        source_file_h5 = os.path.join(combinedPath, h5_path,f'{fileInts}.h5')
        
        destination_path_image = os.path.join(destination_path, trainPath, image_path)
        destination_path_mat = os.path.join(destination_path, trainPath, mat_path)
        destination_path_h5 = os.path.join(destination_path, trainPath, h5_path)
        os.makedirs(destination_path_image, exist_ok=True)
        os.makedirs(destination_path_mat, exist_ok=True)
        os.makedirs(destination_path_h5, exist_ok=True)
        shutil.copy2(source_file_image, destination_path_image)
        shutil.copy2(source_file_mat, destination_path_mat)
        shutil.copy2(source_file_h5, destination_path_h5)

    for fileInts in test_files_index:
        source_file_image = os.path.join(combinedPath, image_path,f'{fileInts}.jpg')
        source_file_mat = os.path.join(combinedPath, mat_path,f'{fileInts}.mat')
        source_file_h5 = os.path.join(combinedPath, h5_path,f'{fileInts}.h5')
        
        destination_path_image = os.path.join(destination_path, testPath, image_path)
        destination_path_mat = os.path.join(destination_path, testPath, mat_path)
        destination_path_h5 = os.path.join(destination_path, testPath, h5_path)
        os.makedirs(destination_path_image, exist_ok=True)
        os.makedirs(destination_path_mat, exist_ok=True)
        os.makedirs(destination_path_h5, exist_ok=True)
        shutil.copy2(source_file_image, destination_path_image)
        shutil.copy2(source_file_mat, destination_path_mat)
        shutil.copy2(source_file_h5, destination_path_h5)

DownloadKaggleDataset()
CombineDatasetParts()
ResplitDataset()
