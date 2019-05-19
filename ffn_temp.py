import cv2
import numpy as np
#import matplotlib.pyplot as plt


# In[3]:
print("importing DL libraries")

import tensorflow.keras as keras
from tensorflow.keras import layers



# X_1=np.load('NN2_inp.npy')
# X_2=np.load('NN1_out.npy')

print("creating model")
inp_rgb=layers.Input(shape=(320,400,3),name='input_rgb')
inp_mask=layers.Input(shape=(320,400,3),name='input_mask')

concat_layer= layers.Concatenate(axis=1)([inp_rgb,inp_mask])

concat_layer= layers.Concatenate(axis=2)([concat_layer,concat_layer])
conv_layer1=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer1', padding='same')(concat_layer)
conv_layer2=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer2', padding='same')(conv_layer1)


# add1=layers.Add()([ concat_layer,conv_layer2 ])
conv_layer3=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer3', padding='same')(conv_layer2)
conv_layer4=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer4', padding='same')(conv_layer3)

add2=layers.Add()([ conv_layer2,conv_layer4 ])
conv_layer5=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer5', padding='same')(add2)
conv_layer6=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer6', padding='same')(conv_layer5)
X_2
add3=layers.Add()([ conv_layer4,conv_layer6 ])
conv_layer7=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer7', padding='same')(add3)
conv_layer8=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer8', padding='same')(conv_layer7)

add4=layers.Add()([ conv_layer6,conv_layer8 ])
conv_layer9=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer9', padding='same')(add4)
conv_layer10=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer10', padding='same')(conv_layer9)

add5=layers.Add()([conv_layer8,conv_layer10])
conv_layer11=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',strides=(2,2),name='conv_layer11', padding='same')(add5)
conv_net=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_net', padding='same')(conv_layer11)

o=layers.Conv2D(filters=1,kernel_size=(3,3),activation='sigmoid',name='conv3x3_output', padding='same')(conv_net)

model = keras.models.Model(inputs=[inp_rgb,inp_mask],outputs=o)

model.summary()