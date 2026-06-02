import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import scipy.io as spio
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from attention import attention
from scipy.stats import zscore
import timeit
start = timeit.default_timer()
#%%
def custom_loss(Y_true, Y_pred):
    PSNR = tf.image.psnr(Y_true, Y_pred, max_val = 255)
    return PSNR

#%%
data = spio.loadmat("Botswana.mat")["Botswana"]   
#data_gt = spio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
data_norm = zscore(data)
#data_norm = np.stack((data_norm, data_norm))
data_norm = np.reshape(data_norm, (1,1476,256,145))
data = data.reshape(1,1476,256,145)
#%%
#from spectral import *
import tifffile
#tifffile.imshow(data_norm[:,:,:,7])
#tifffile.imshow(data)
#tifffile.imshow(data_gen)


#%%
opt = Adam(learning_rate=0.0009, beta_1=0.9,beta_2 = 0.999)
encoder = Sequential()

#encoder.add(Conv2D(220,( 3, 3),activation = 'relu'))  
#encoder.add(MaxPooling2D(pool_size = (2, 2))) 

encoder.add(Conv2D(128,( 3, 3),activation = 'relu'))  
encoder.add(MaxPooling2D(pool_size = (2, 2))) 
encoder.add(Conv2D(64,( 3, 3), activation = 'relu'))  
encoder.add(MaxPooling2D(pool_size = (2, 2)))  
#encoder.add(tf.keras.layers.Reshape((34,34,64)))
encoder.add(Conv2D(32,( 3, 3), activation = 'relu'))  
encoder.add(MaxPooling2D(pool_size = (2, 2)))
encoder.add(attention())
#encoder.add(attention())
#encoder.add(Flatten())
#encoder.add(Dense(units = 8192,activation = 'relu'))   
encoder.compile(optimizer=opt)

decoder = Sequential()
#decoder.add(tf.keras.layers.Reshape((16,16,32)))
decoder.add(Flatten())
decoder.add(Dense(units = 30000, activation = 'relu'))

decoder.add(tf.keras.layers.Reshape((100,20,15)))
decoder.add(UpSampling2D(size = (2,2)))
decoder.add(Conv2DTranspose(32,(3, 3), activation = "relu"))    
decoder.add(UpSampling2D(size = (2,2)))
decoder.add(Conv2DTranspose(64,(3, 3), activation = "relu"))
decoder.add(UpSampling2D(size = (2,2)))

decoder.add(Conv2DTranspose(128,(3, 3), activation = "relu"))
#decoder.add(UpSampling2D(size = (2,2)))

decoder.add(Conv2DTranspose(145,(3,3), activation = "relu"))
decoder.add(Conv2DTranspose(145,(3,3), activation = "linear"))
decoder.add(Resizing(1476,256))
decoder.compile(optimizer=opt)
    
model = Sequential()
model.add(encoder)
model.add(decoder)

model.compile(optimizer = opt, metrics = custom_loss, loss = "mse")

model.fit(data_norm, data_norm, epochs = 350)
    
data_gen = model.predict(data_norm)

print(custom_loss(data_norm, data_gen))

#model.save("IP_try2.h5")
#encoder.save("IP_try2_enc.h5")
#decoder.save("IP_try2_dec.h5")
data_enc = encoder.predict(data_norm)
tifffile.imshow(data_gen[:,:,:,7])
tifffile.imshow(data_norm[:,:,:,7])




stop = timeit.default_timer()
print("time: ",stop -start)


#%%
import os
tifffile.imsave("data_enc5.tiff", data_enc)
stats_enc = os.stat("data_enc5.tiff")
stats_data = os.stat("Botswana.mat")    
print(stats_data.st_size/stats_enc.st_size)


#%%%





