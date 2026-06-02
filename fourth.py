import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as spio
import scipy.stats
import tensorflow
from PIL import Image
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
import math
from scipy.stats import zscore
import timeit
start = timeit.default_timer()
#%%
def custom_loss(Y_true, Y_pred):
    PSNR = tf.image.psnr(Y_true, Y_pred, max_val = 255)
    return PSNR

#%%
data4 = spio.loadmat("KSC.mat")["KSC"]   
data_gt4 = spio.loadmat('KSC_gt.mat')['KSC_gt']

data_norm4 = zscore(data4)
#data_norm = np.stack((data_norm, data_norm))
data_norm4 = np.reshape(data_norm4, (1,512,614,176))

#%%
from spectral import *
import tifffile
#tifffile.imshow(data_norm4)
#tifffile.imshow(data4)
#tifffile.imshow(data_gen4)
data_norm4.size
#%%
opt = Adam(lr=0.0009, beta_1=0.9,beta_2 = 0.999)
encoder4 = Sequential()

encoder4.add(Conv2D(128,( 3, 3),activation = 'relu'))  
encoder4.add(MaxPooling2D(pool_size = (2, 2))) 
encoder4.add(Conv2D(64,( 3, 3), activation = 'relu'))  
encoder4.add(MaxPooling2D(pool_size = (2, 2)))  
encoder4.add(Conv2D(32,( 3, 3), activation = 'relu'))  
encoder4.add(MaxPooling2D(pool_size = (2, 2)))   
encoder4.add(attention())  
encoder4.add(Flatten())
#encoder4.add(Dense(units = 1200,activation = 'relu')) 
encoder4.compile(optimizer=opt)

decoder4 = Sequential()
decoder4.add(Dense(units = 148800))
decoder4.add(tf.keras.layers.Reshape((62,75,32)))
#decoder4.add(tf.keras.layers.Reshape((15,20,8)))
#decoder4.add(UpSampling2D(size = (2,2)))
decoder4.add(Conv2DTranspose(64,(3, 3), activation = "relu"))    
decoder4.add(UpSampling2D(size = (2,2)))
decoder4.add(Conv2DTranspose(64,(3, 3), activation = "relu"))
decoder4.add(UpSampling2D(size = (2,2)))

decoder4.add(Conv2DTranspose(128,(3, 3), activation = "relu"))
decoder4.add(UpSampling2D(size = (2,2)))

decoder4.add(Conv2DTranspose(176,(3,3), activation = "relu"))
decoder4.add(Conv2DTranspose(176,(3,3), activation = "linear"))
decoder4.add(Resizing(512,614))
decoder4.compile(optimizer=opt)
    
model4 = Sequential()
model4.add(encoder4)
model4.add(decoder4)

model4.compile(optimizer = opt, metrics = custom_loss, loss = "mse")

model4.fit(data_norm4, data_norm4, epochs = 350)
    
data_gen4 = model4.predict(data_norm4)

print(custom_loss(data_norm4, data_gen4))

#model4.save("ksc_try1.h5")
#encoder4.save("ksc_try1_enc.h5")
#decoder4.save("ksc_try1_dec.h5")
data_enc4 = encoder4.predict(data_norm4)
tifffile.imshow(data_gen4[:,:,:,7])
stop = timeit.default_timer()
print("time: ",stop-start)
#%%
import os
tifffile.imsave("data_enc4.tiff", data_enc4)
stats_enc = os.stat("data_enc4.tiff")
stats_data = os.stat("KSC.mat")    
print(stats_data.st_size/stats_enc.st_size)




