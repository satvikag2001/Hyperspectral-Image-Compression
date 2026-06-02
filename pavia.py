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
data3 = spio.loadmat("PaviaU.mat")["paviaU"]   
#data_gt3 = spio.loadmat('PaviaU_gt.mat')['paviaU_gt']

data_norm3 = zscore(data3)
data_norm3 = np.reshape(data_norm3,(1, 610, 340, 103))
#water_bands3 = [108,109,110,111,112,154,155,156,157,158,159,160,161,162,163,164,165,166,167,224,-1]
#data_3 = np.delete(data_norm3,water_bands3,axis = 3)
#%%
import tifffile
#tifffile.imshow(data_norm3[:,:,:,7])
#tifffile.imshow(data3)
#tifffile.imshow(data_gen)

#%%
opt = Adam(lr=0.0009, beta_1=0.9,beta_2 = 0.999)

encoder3 = Sequential()

#encoder2.add(Conv2D(103,( 3, 3),activation = 'relu'))  
#encoder2.add(MaxPooling2D(pool_size = (2, 2))) 
encoder3.add(Conv2D(64,( 3, 3), activation = 'relu'))  
encoder3.add(MaxPooling2D(pool_size = (2, 2)))  
encoder3.add(Conv2D(32,( 3, 3), activation = 'relu'))  
encoder3.add(MaxPooling2D(pool_size = (2, 2))) 
encoder3.add(attention())  

encoder3.add(Flatten())
#encoder3.add(tf.keras.layers.Reshape((401056,1)))

#encoder3.add(tf.keras.layers.Reshape((151,83,32)))

#encoder3.add(Dense(units = 12000,activation = 'relu'))   

encoder3.compile(optimizer=opt)

decoder3 = Sequential()
decoder3.add(Dense(units = 100000, activation = "relu"))
decoder3.add(tf.keras.layers.Reshape((100,50,20)))
#decoder3.add(Conv2DTranspose(64,(3, 3), activation = "relu"))    
#decoder3.add(UpSampling2D(size = (2,2)))

decoder3.add(UpSampling2D(size = (2,2)))
decoder3.add(Conv2DTranspose(64,(3, 3), activation = "relu"))    
decoder3.add(UpSampling2D(size = (2,2)))
decoder3.add(Conv2DTranspose(64,(3, 3), activation = "relu"))
decoder3.add(UpSampling2D(size = (2,2)))

#decoder3.add(Conv2DTranspose(128,(3, 3), activation = "relu"))

decoder3.add(Conv2DTranspose(103,(3,3), activation = "relu"))
#decoder3.add(UpSampling2D(size = (2,2)))

decoder3.add(Conv2DTranspose(103,(3,3), activation = "linear"))
decoder3.add(Resizing(610,340))
decoder3.compile(optimizer=opt)
    
model3 = Sequential()
model3.add(encoder3)
model3.add(decoder3)

model3.compile(optimizer = opt, metrics = custom_loss, loss = "mse")

model3.fit(data_norm3, data_norm3, epochs = 350)
    
data_gen3 = model3.predict(data_norm3)

print(custom_loss(data_norm3, data_gen3))

#model3.save("paviaU_try1.h5")
#encoder3.save("paviaU_try1_enc.h5")
#decoder3.save("paviaU_try1_dec.h5")

data_enc3 = encoder3.predict(data_norm3)
tifffile.imshow(data_gen3[:,:,:,0])
tifffile.imshow(data_norm3[:,:,:,0])


stop = timeit.default_timer()
print("time: ",stop -start)
#%%
import os
tifffile.imsave("data_enc3.tiff", data_enc3)
stats_enc = os.stat("data_enc3.tiff")
stats_data = os.stat("PaviaU.mat")    
print(stats_data.st_size/stats_enc.st_size)

