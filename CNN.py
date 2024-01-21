# File for functions of CNN
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Conv2DTranspose
from keras.optimizers import Adam
import matplotlib.pyplot as plt, numpy as np
from sklearn.model_selection import train_test_split
from dataset import trainImgH5

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(180, 180, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Flatten and fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(180, activation='softmax'))  

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae'])
model.summary()

lr_monitor = LearningRateScheduler(lambda epochs : 1e-8 * 10 ** (epochs/20))
img, labels = trainImgH5(trainortest='combined')
# x_test, y_test = trainImgH5(trainortest='test')
print('got data',img.shape,labels.shape)

x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1)
x_train, x_test = x_train / 255.0, x_test / 255.0

####### copied from kaggle ######
figure,axis = plt.subplots(1,2,figsize=(10,10))

axis[0].imshow(y_train[1],cmap="jet")
axis[0].set_xlabel(y_train[1].shape)
axis[0].set_title("MAT")
axis[1].imshow(x_train[1])
axis[1].set_xlabel(x_train[1].shape)
axis[1].set_title("ORIGINAL")
plt.show()

Early_Stopper = EarlyStopping(monitor="loss",patience=3,mode="min")
Checkpoint_Model = ModelCheckpoint(monitor="val_accuracy",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      filepath="./modelcheck")
compile_loss = "mse"
compile_optimizer = Adam(lr=0.0000001)
output_class = 1

Encoder_AE = Sequential()
Encoder_AE.add(Conv2D(32,(2,2),kernel_initializer = 'he_normal'))
Encoder_AE.add(BatchNormalization())
Encoder_AE.add(ReLU())
#
Encoder_AE.add(Conv2D(64,(2,2),kernel_initializer = 'he_normal'))
Encoder_AE.add(BatchNormalization())
Encoder_AE.add(ReLU())
#
Encoder_AE.add(Conv2D(128,(2,2),kernel_initializer = 'he_normal'))
Encoder_AE.add(BatchNormalization())
Encoder_AE.add(ReLU())
#
Encoder_AE.add(Conv2D(256,(2,2),kernel_initializer = 'he_normal'))
Encoder_AE.add(BatchNormalization())
Encoder_AE.add(ReLU())

Decoder_AE = Sequential()
Decoder_AE.add(Conv2DTranspose(128,(2,2)))
Decoder_AE.add(ReLU())
#
Decoder_AE.add(Conv2DTranspose(64,(2,2)))
Decoder_AE.add(ReLU())
#
Decoder_AE.add(Conv2DTranspose(32,(2,2)))
Decoder_AE.add(ReLU())
#
Decoder_AE.add(Conv2DTranspose(output_class,(2,2)))
Decoder_AE.add(ReLU())

Auto_Encoder = Sequential([Encoder_AE,Decoder_AE])
Auto_Encoder.compile(loss=compile_loss,optimizer=compile_optimizer)

Model_AutoEncoder_History = Auto_Encoder.fit(x_train, y_train,epochs=3,validation_data=(x_val,y_val),callbacks=[Early_Stopper,Checkpoint_Model])

Prediction_Seen = Auto_Encoder.predict(x_test[:5])
figure,axis = plt.subplots(1,3,figsize=(10,10))
axis[0].imshow(x_test[0])
axis[0].set_title("ORIGINAL")
axis[1].imshow(Prediction_Seen[0],cmap="jet")
axis[1].set_title("PREDICTION")
axis[2].imshow(y_test[0],cmap="jet")
axis[2].set_title("TRUE")
plt.show()
figure,axis = plt.subplots(1,3,figsize=(10,10))
axis[0].imshow(x_test[1])
axis[0].set_title("ORIGINAL")
axis[1].imshow(Prediction_Seen[1],cmap="jet")
axis[1].set_title("PREDICTION")
axis[2].imshow(y_test[1],cmap="jet")
axis[2].set_title("TRUE")
plt.show()
figure,axis = plt.subplots(1,3,figsize=(10,10))
axis[0].imshow(x_test[2])
axis[0].set_title("ORIGINAL")
axis[1].imshow(Prediction_Seen[2],cmap="jet")
axis[1].set_title("PREDICTION")
axis[2].imshow(y_test[2],cmap="jet")
axis[2].set_title("TRUE")
plt.show()
figure,axis = plt.subplots(1,3,figsize=(10,10))
axis[0].imshow(x_test[3])
axis[0].set_title("ORIGINAL")
axis[1].imshow(Prediction_Seen[3],cmap="jet")
axis[1].set_title("PREDICTION")
axis[2].imshow(y_test[3],cmap="jet")
axis[2].set_title("TRUE")
plt.show()
figure,axis = plt.subplots(1,3,figsize=(10,10))
axis[0].imshow(x_test[4])
axis[0].set_title("ORIGINAL")
axis[1].imshow(Prediction_Seen[4],cmap="jet")
axis[1].set_title("PREDICTION")
axis[2].imshow(y_test[4],cmap="jet")
axis[2].set_title("TRUE")
plt.show()
print(np.count_nonzero(Prediction_Seen[0]))
print(np.count_nonzero(Prediction_Seen[1]))
print(np.count_nonzero(Prediction_Seen[2]))
print(np.count_nonzero(Prediction_Seen[3]))
print(np.count_nonzero(Prediction_Seen[4]))


############
# history = model.fit(x_train, y_train, epochs=50, batch_size=32)

# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.axis([np.min(history.history['lr']), np.max(history.history['lr']), np.min(history.history['loss']), 15])
# plt.show()