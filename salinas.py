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
data2 = spio.loadmat("Salinas.mat")["salinas"]   
data_gt2 = spio.loadmat('Salinas_gt.mat')['salinas_gt']

data_norm2 = zscore(data2)
data_norm2 = np.reshape(data_norm2,(1, 512, 217, 224))
#water_bands2 = [108,109,110,111,112,154,155,156,157,158,159,160,161,162,163,164,165,166,167,224,-1]
#data_p = np.delete(data_norm2,water_bands2,axis = 3)
#%%
from spectral import *
import tifffile
#tifffile.imshow(data_norm2[:,:,:,7])
#tifffile.imshow(data2)
#tifffile.imshow(data_gen)
        
#%%
opt = Adam(lr=0.0009, beta_1=0.9,beta_2 = 0.999)

encoder2 = Sequential()

encoder2.add(Conv2D(128,( 3, 3),activation = 'relu'))  
encoder2.add(MaxPooling2D(pool_size = (2, 2))) 
encoder2.add(Conv2D(64,( 3, 3), activation = 'relu'))  
encoder2.add(MaxPooling2D(pool_size = (2, 2)))  
encoder2.add(Conv2D(32,( 3, 3), activation = 'relu'))  
encoder2.add(MaxPooling2D(pool_size = (2, 2)))   

encoder2.add(attention())
encoder2.add(Flatten())
  

encoder2.compile(optimizer=opt)

decoder2 = Sequential()
decoder2.add(Dense(units = 49600, activation = "relu"))
decoder2.add(tf.keras.layers.Reshape((62,25,32)))
decoder2.add(UpSampling2D(size = (2,2)))
decoder2.add(Conv2DTranspose(64,(3, 3), activation = "relu"))    
decoder2.add(UpSampling2D(size = (2,2)))
decoder2.add(Conv2DTranspose(64,(3, 3), activation = "relu"))
decoder2.add(UpSampling2D(size = (2,2)))

decoder2.add(Conv2DTranspose(128,(3, 3), activation = "relu"))

decoder2.add(Conv2DTranspose(224,(3,3), activation = "relu"))
decoder2.add(Conv2DTranspose(224,(3,3), activation = "linear"))
decoder2.add(Resizing(512,217))
decoder2.compile(optimizer=opt)
    
model2 = Sequential()
model2.add(encoder2)
model2.add(decoder2)

model2.compile(optimizer = opt, metrics = custom_loss, loss = "mse")

model2.fit(data_norm2, data_norm2, epochs = 350)
    
data_gen2 = model2.predict(data_norm2)

print(custom_loss(data_norm2, data_gen2))

#model2.save("salinas_try2.h5")
#encoder2.save("salinas_try2_enc.h5")
#   decoder2.save("salinas_try2_dec.h5")

data_enc2 = encoder2.predict(data_norm2)
#tifffile.imshow(data_gen2)
#tifffile.imshow(data_norm2[:,:,:,0])
#tifffile.imshow(data2[:,:,0])
#tifffile.imshow(data_gen2[:,:,:,0])

stop = timeit.default_timer()
print("time: ",stop -start)

'''
#%%
import os
tifffile.imsave("data_enc2.tiff", data_enc2)
stats_enc = os.stat("data_enc2.tiff")
stats_data = os.stat("Salinas.mat")    
print(stats_data.st_size/stats_enc.st_size)
'''
