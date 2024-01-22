# File for functions of CNN
import tensorflow as tf
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Conv2DTranspose
from keras.optimizers import Adam
import matplotlib.pyplot as plt, numpy as np
from sklearn.model_selection import train_test_split
from dataset import trainImgH5

img, labels = trainImgH5(trainortest='combined')
print('got data',img.shape,labels.shape)
x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1)
x_train, x_test = x_train / 255.0, x_test / 255.0



modelPath = "./modelcheck.h5"

Checkpoint_Model = ModelCheckpoint(monitor="val_mean_squared_error",
                                   save_best_only=True,
                                   save_weights_only=True,
                                   mode="min",
                                   filepath=modelPath)
Early_Stopper = EarlyStopping(monitor="val_mean_squared_error",
                              patience=10,
                              mode="min",
                              restore_best_weights=True)

print(Checkpoint_Model)
print(Early_Stopper)

compile_loss = "mean_squared_error"
compile_optimizer = Adam(lr=0.00001)
output_class = 1

kernel=(3,3)
# Custom model layers
model1 = Sequential()
model1.add(Conv2D(64, kernel_size = kernel, input_shape = (180,180,3),activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(strides=2))
model1.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(strides=2))
model1.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(strides=2))            
model1.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
model1.add(BatchNormalization())
model1.add(Flatten())
model1.add(Dense(1024, activation='relu'))
model1.add(Dense(1, activation="linear"))

# Display the combined model summary
model1.summary()

model1.compile(
    optimizer=compile_optimizer, 
    loss=compile_loss, 
    metrics=['mean_absolute_error', 'mean_squared_error']
)

model1.fit(x_train, y_train,epochs=30,validation_data=(x_val,y_val),callbacks=[Early_Stopper,Checkpoint_Model])
print('\nModel Training Finished.')

