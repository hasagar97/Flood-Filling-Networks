import cv2
import numpy as np
#import matplotlib.pyplot as plt


# In[3]:
print("importing DL libraries")

import tensorflow.keras as keras
from tensorflow.keras import layers

def image_generator(x_1,x_2,y, batch_size = 64):
	count= -batch_size
	length=x_1.shape[0]
	y=np.expand_dims(y, axis=3)
	while True:
		# Select files (paths/indices) for the batch
		count+=batch_size
		if(count>=length- batch_size):
			count=0
		batch_input_1 = []
		batch_input_2 = []
		batch_output = [] 

		# Read in each input, perform preprocessing and get labels
		for i in range(batch_size):
			input_1 = x_1[i+count]
			input_2 = x_2[i+count]
			output = y[i+count]

			# input = preprocess_input(image=input)
			batch_input_1 += [ input_1 ]
			batch_input_2 += [ input_2 ]
			batch_output += [ output ]
		# Return a tuple of (input,output) to feed the network
		batch_x_1 = np.array( batch_input_1 )
		batch_x_2 = np.array( batch_input_2 )
		batch_y = np.array( batch_output )

		yield( [batch_x_1,batch_x_2], batch_y )

print("reading data")
X_1=np.load('nn2_inp_img5000.npy')/255
X_2=np.load('nn2_inp_mask5000.npy')
Y=np.load('nn2_out_mask5000.npy')/255

print(X_1.dtype,X_1[0][1][2])
print(X_2.dtype,X_2[0][1][2])
print(Y.dtype,Y[0][1][2])

print("X_1",X_1.shape)
print("X_2",X_2.shape)
print("Y",Y.shape)

print("creating model")
inp_rgb=layers.Input(shape=(320,400,3),name='input_rgb')
inp_mask=layers.Input(shape=(320,400,1),name='input_mask')

concat_layer= layers.Concatenate()([inp_rgb,inp_mask])

conv_layer1=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer1', padding='same')(concat_layer)
conv_layer2=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer2', padding='same')(conv_layer1)


# add1=layers.Add()([ concat_layer,conv_layer2 ])
conv_layer3=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer3', padding='same')(conv_layer2)
conv_layer4=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer4', padding='same')(conv_layer3)

add2=layers.Add()([ conv_layer2,conv_layer4 ])
conv_layer5=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer5', padding='same')(add2)
conv_layer6=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer6', padding='same')(conv_layer5)

add3=layers.Add()([ conv_layer4,conv_layer6 ])
conv_layer7=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer7', padding='same')(add3)
conv_layer8=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer8', padding='same')(conv_layer7)

add4=layers.Add()([ conv_layer6,conv_layer8 ])
conv_layer9=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer9', padding='same')(add4)
conv_layer10=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer10', padding='same')(conv_layer9)

add5=layers.Add()([conv_layer8,conv_layer10])
conv_layer11=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_layer11', padding='same')(add5)
conv_net=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv_net', padding='same')(conv_layer11)

o=layers.Conv2D(filters=1,kernel_size=(3,3),activation='sigmoid',name='conv3x3_output', padding='same')(conv_net)

model = keras.models.Model(inputs=[inp_rgb,inp_mask],outputs=o)

model.summary()




model.compile(loss='binary_crossentropy',
      optimizer= 'Adam' ,
      metrics=['accuracy'])


bs=8
trainGen=image_generator(X_1[:4000],X_2[:4000],Y[:4000], batch_size = bs)
print("batch Size",bs)
H = model.fit_generator(trainGen,steps_per_epoch=int(X_1[:4000].shape[0]/bs)	,epochs=4)
model.save('ffn_4000.h5')

# In[16]:


# print(yyy.shape)
score = model.evaluate([X_1[4000:],X_2[4000:]],np.expand_dims(Y[4000:], axis=3) )


# In[34]:


print(score)


# model.save('ffn_'+str(x_train.shape[0])+'.h5')


y_pred = model.predict([X_1[4000:],X_2[4000:]])
np.save('ffn_out',y_pred)


