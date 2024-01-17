# File for functions for dataset
# from kaggle.api.kaggle_api_extended import KaggleApi
import os, shutil

# Initialize Kaggle API
# api = KaggleApi()
# api.authenticate()


# dataset_name = 'tthien/shanghaitech-with-people-density-map'
# api.dataset_download_files(dataset_name, path='./shanghaitech_data', unzip=True)

# setup code from https://www.kaggle.com/code/donkeys/kaggle-python-api

# Combine Part A & B
partAdir = './shanghaitech_data/ShanghaiTech/part_A/'
partBdir = './shanghaitech_data/ShanghaiTech/part_B/'
dataSplit = ['train_data','test_data'] 
finalDirLevel = ['ground-truth','ground-truth-m5','images']

combinedDir = './shanghaitech_data/ShanghaiTechCombined/'

if not os.path.exists(combinedDir):
        os.makedirs(combinedDir)

for split in ['train_data','test_data']:
    # Iterate through images, gt, and gt-m5 folders
    for folder in ['ground-truth','ground-truth-h5','images']:
        print(split,folder)
        counter = 0
        source_a_path = os.path.join(partAdir, split, folder)  # Modified this line
        source_b_path = os.path.join(partBdir, split, folder)  # Modified this line
        destination_path = os.path.join(combinedDir, folder)  # Modified this line
        print(source_a_path)
        print(source_b_path)
        print(repr(source_a_path))
        print(repr(source_b_path))

        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        for partPath in [source_a_path, source_b_path]:
            print(partPath)
            print(repr(partPath))
            for filename in os.listdir(partPath):
                source_file_path = os.path.join(partPath, filename)
                destination_file_path = os.path.join(destination_path, f'{counter}.{filename.split(".")[-1]}')  # Modified this line
                counter += 1
                shutil.copy2(source_file_path, destination_file_path)


