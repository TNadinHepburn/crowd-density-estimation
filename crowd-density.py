# Main file for crowd density estimation
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("python main function")

    imagepath = '.\\shanghaitech_data\\ShanghaiTechCombined\\images\\0.jpg'
    imageread = cv2.cvtColor(cv2.imread(imagepath),cv2.COLOR_BGR2RGB)
    imageread2 = cv2.cvtColor(cv2.imread(imagepath),cv2.COLOR_BGR2RGB)
    # figure = plt.figure(figsize=(10,10))
    # plt.imshow(imageread)
    # plt.show()

    # Load the .mat file
    mat_data = loadmat('.\\shanghaitech_data\\ShanghaiTechCombined\\ground-truth\\0.mat')
    #print(type(mat_data))
    #print(mat_data.items())
    #print(mat_data.keys())
    

    # Extract the image data, assuming it's stored in a variable named 'image_data'
    image_data = mat_data["image_info"]
    #print(type(image_data))

    coords = image_data[0][0][0][0][0]
    print(coords)
    print(type(coords))
    print(coords.shape)
    # Display the image

    figure = plt.figure(figsize=(10,10))

    for e_x_cor, e_y_cor in coords:
        e_x_cor = int(e_x_cor)
        e_y_cor = int(e_y_cor)
        cv2.drawMarker(imageread, (e_x_cor, e_y_cor), (255, 0, 0),thickness=1)

    plt.imshow(imageread)
    plt.title("testing")
    
    # plt.imshow(image_data, cmap='jet')  # Adjust the cmap based on your data
    plt.show()

    Example_Testing_Zeros = np.zeros((imageread2.shape[0], imageread2.shape[1]), dtype=np.float32)

    for x_cor_example, y_cor_example in coords:
        x_cor_example = int(x_cor_example)
        y_cor_example = int(y_cor_example)
        Example_Testing_Zeros[y_cor_example, x_cor_example] = 1
        
    figure = plt.figure(figsize=(10,10))
    plt.imshow(Example_Testing_Zeros,cmap="gray")
    plt.show()

    Gaussian_Image_Example = gaussian_filter(Example_Testing_Zeros,sigma=5,truncate=5*5)
    figure = plt.figure(figsize=(10,10))
    plt.xlabel(Gaussian_Image_Example.shape)
    plt.imshow(Gaussian_Image_Example,cmap="jet")
    plt.show()


if __name__ == '__main__':
    main()