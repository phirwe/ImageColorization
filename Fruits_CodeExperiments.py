
# coding: utf-8

# In[ ]:




######################################################

'''

This model is trained for 1000 epochs on April 21 on scenes.

'''

######################################################





# In[1]:

from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Conv3D, UpSampling3D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image
import PIL


# In[2]:

# Get imagesimg.save('/Users/poorwa/torch/siggraph2016_colorization/traintrynew.jpg')

X = []
for filename in os.listdir('Dataset/blackberries/blackberries/Train_256/'):
        ext = filename[filename.index(filename[:-3][-1])+1:]
        print ('here')
        img = Image.open('Dataset/blackberries/blackberries/Train_256/'+filename)
        #img = img.resize((256, 256), PIL.Image.ANTIALIAS)
        #img.save('Train/'+filename)
        X.append(img_to_array(load_img('Dataset/blackberries/blackberries/Train_256/'+filename)))
X = np.array(X, dtype=float)



# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain


# In[3]:

# model = Sequential()
# model.add(InputLayer(input_shape=(None, None, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2,2)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2,2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
# model.add(UpSampling2D((2,2)))
# model.compile(optimizer='rmsprop', loss='mse')

# model = Sequential()
# model.add(InputLayer(input_shape=(256, 256, 1)))
# model.add(Conv2D(64, (129, 129), activation='relu', padding='valid'))
# # model.add(Conv2D(64, (129, 129), activation='relu', padding='valid', strides=1))
# model.add(Conv2D(128, (65, 65), activation='relu', padding='valid'))
# # model.add(Conv2D(128, (65, 65), activation='relu', padding='valid', strides=2))
# model.add(Conv2D(256, (33, 33), activation='relu', padding='valid'))
# # model.add(Conv2D(256, (33, 33), activation='relu', padding='valid', strides=2))
# model.add(Conv2D(512, (1, 1), activation='relu', padding='valid'))
# # model.add(Conv2D(512, (1, 1), activation='relu', padding='valid'))
# # model.add(Conv2D(512, (1, 1), activation='relu', padding='valid'))
# # model.add(Conv2D(512, (1, 1), activation='relu', padding='valid'))
# # model.add(UpSampling2D((2,2)))
# model.add(Conv2D(256, (1, 1), activation='relu', padding='valid'))
# model.add(UpSampling2D((2,2)))
# model.add(UpSampling2D((2,2)))
# model.add(UpSampling2D((2,2)))
# model.add(Conv2D(2, (128, 128), activation='relu', padding='same'))
# model.compile(optimizer='rmsprop', loss='mse')
# model.summary()


# model = Sequential()
# model.add(InputLayer(input_shape=(None, None, 1)))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (1, 1), activation='relu', padding='valid'))
# model.add(Conv2D(64, (1, 1), activation='relu', padding='valid'))
# model.add(Conv2D(32, (1, 1), activation='relu', padding='valid'))
# model.add(Conv2D(16, (1, 1), activation='relu', padding='valid'))
# model.add(Conv2D(8, (1, 1), activation='relu', padding='valid'))
# model.add(Conv2D(4, (1, 1), activation='relu', padding='valid'))
# model.add(Conv2D(2, (1, 1), activation='relu', padding='valid'))

# model.compile(optimizer='rmsprop', loss='mse')

model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.compile(optimizer='rmsprop', loss='mse')
model.summary()


# In[ ]:

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
batch_size = 10
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
#         print("1. ", X_batch.shape)
#         print("2. ", Y_batch.shape)
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Train model      
tensorboard = TensorBoard(log_dir="first_run")
model.fit_generator(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=500, steps_per_epoch=10)


# In[ ]:

# Save model
model_json = model.to_json()
with open("April21_1000_face.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("April21_1000_face.h5")


# In[ ]:

# Test images
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=batch_size))


# In[ ]:
filenames = []
color_me = []
for filename in os.listdir('Dataset/blackberries/blackberries/Test_256/'):
    filenames.append(filename)
    color_me.append(img_to_array(load_img('Dataset/blackberries/blackberries/Test_256/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("Dataset/blackberries/blackberries/Test_256/colored_"+filenames[i], lab2rgb(cur))


# In[ ]:

# for filename in os.listdir('Test/'):
#     img = Image.open('Test/'+filename)
#     img = img.resize((256, 256), PIL.Image.ANTIALIAS)
#     img = img.convert('LA')
#     img.save('Test/'+filename[:-3]+'.png')


# In[11]:




# In[ ]:



