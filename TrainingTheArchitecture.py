
import cv2
import numpy as np
#import matplotlib.pyplot as plt


# In[3]:
print("importing DL libraries")

import tensorflow.keras as keras
from tensorflow.keras import layers


# In[4]:

print("reading Daata")
X=np.load('data.npy')
Y=np.load('mask.npy')
print("done reading")

# In[5]:


x_train=X[:5000]
x_test=X[5000:]
y_train=Y[:5000]
y_test=Y[5000:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = y_train.astype('float32') / 255
y_test = y_test.astype('float32') / 255

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
del X
del Y


def image_generator_NN1(x_train,y_train, batch_size = 64):
	count= -64
	length=x_train.shape[0]
	y_train=np.expand_dims(y_train, axis=3)
	while True:
		# Select files (paths/indices) for the batch
		count+=64
		if(count>=length-64):
			count=0
		batch_input = []
		batch_output = [] 

		# Read in each input, perform preprocessing and get labels
		for i in range(batch_size):
			input = x_train[i+count]
			output = y_train[i+count]

			# input = preprocess_input(image=input)
			batch_input += [ input ]
			batch_output += [ output ]
		# Return a tuple of (input,output) to feed the network
		batch_x = np.array( batch_input )
		batch_y = np.array( batch_output )

		yield( batch_x, batch_y )

#Defining Model
inp=layers.Input(shape=(320,400,3),name='input_layer')
en1 = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',name='conv3x3_en1', padding='same')(inp)
en1_1 = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',name='conv3x3_en1_1', padding='same')(en1)
en1_max = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool') (en1_1)

en2 = layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv3x3_en2', padding='same')(en1_max)
en2_1 = layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv3x3_en2_1', padding='same')(en2)
en2_max = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool') (en2_1)

en3 = layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',name='conv3x3_en3', padding='same')(en2_max)
en3_1 = layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',name='conv3x3_en3_1', padding='same')(en3)
en3_max = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool') (en3_1)
print(en3_max.shape)

en4=layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',name='conv3x3_en4', padding='same')(en3_max)
en4_1=layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',name='conv3x3_en4_1', padding='same')(en4)
en4_max = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool') (en4_1)
print(en4_max.shape)
m=en4_max

de1=layers.Conv2DTranspose( 128 , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, padding='same',activation='relu'  )(m)
# de2=layers.Conv2DTranspose( 75 , kernel_size=(1,1) , padding='same',activation='relu',data_format='channels_first'  )(de1)
de1_1=layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',name='conv3x3_de1_1', padding='same')(de1)
de1_2=layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',name='conv3x3_de1_2', padding='same')(de1_1)
print(de1_2.shape)

add1=layers.Add()([ de1_2 , en3_max ])

de2=layers.Conv2DTranspose( 64 , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, padding='same',activation='relu'  )(add1)
# de2=layers.Conv2DTranspose( 75 , kernel_size=(1,1) , padding='same',activation='relu',data_format='channels_first'  )(de1)
de2_1=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv3x3_de2_1', padding='same')(de2)
de2_2=layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv3x3_de2_2', padding='same')(de2_1)
add2=layers.Add()([ de2_2 , en2_max ])

de3=layers.Conv2DTranspose( 32 , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, padding='same',activation='relu'  )(add2)
de3_1=layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',name='conv3x3_de3_1', padding='same')(de3)
de3_2=layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',name='conv3x3_de3_2', padding='same')(de3_1)
add3=layers.Add()([de3_2,en1_max])


de4 = layers.Conv2DTranspose( 16 , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, padding='same',activation='relu'  )(add3)
de4_1=layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',name='conv3x3_de4_1', padding='same')(de4)
de4_2=layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',name='conv3x3_de4_2', padding='same')(de4_1)

o=layers.Conv2D(filters=1,kernel_size=(3,3),activation='sigmoid',name='conv3x3_output', padding='same')(de4_2)
print("added layers")
# In[11]:


model_NN1=keras.models.Model(inp,o)


# In[12]:


model_NN1.summary()


# In[13]:


model_NN1.compile(loss='binary_crossentropy',
      optimizer= 'Adam' ,
      metrics=['accuracy'])


# In[14]:
print("compiled model")

# tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)


# In[15]:


yy=np.expand_dims(y_train, axis=3)#(y_train,(9000,300,400,1))


# In[16]:


yy.shape


# In[17]:


y_train.shape


# In[18]:
bs=4
trainGen_NN1=image_generator_NN1(x_train,y_train, batch_size = bs)
print("batch Size",bs)
H = model_NN1.fit_generator(trainGen_NN1,steps_per_epoch=int(x_train.shape[0]/bs)	,epochs=4)
# model.fit(x=x_train,
         # y=yy,
         # batch_size=300,
         # epochs=1,
         # validation_split=0.1,
         # verbose =1,
         # callbacks=[tbCallBack])


# In[19]:


abc=np.zeros((28,28))
abc.shape
abc=np.reshape(abc,(28,28,1))


# In[20]:


abc.shape

yyy=np.expand_dims(y_test, axis=3)

model.save('nn1_'+str(x_train.shape[0])+'.h5')


y_pred = model.predict(x_test)
np.save('nn2_inp_mask'+str(x_train.shape[0]),y_pred)


X_1=x_test
X_2=y_pred
Y=y_test



def image_generator_FFN(x_1,x_2,y, batch_size = 64):
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
trainGen_FFN=image_generator_FFN(X_1,X_2,Y, batch_size = bs)
print("batch Size",bs)
H = model.fit_generator(trainGen_FFN,steps_per_epoch=int(X_1[:4000].shape[0]/bs)	,epochs=4)
model.save('ffn_4000.h5')

# In[16]:


# print(yyy.shape)
score = model.evaluate([X_1[4000:],X_2[4000:]],np.expand_dims(Y[4000:], axis=3) )


# In[34]:


print(score)


# model.save('ffn_'+str(x_train.shape[0])+'.h5')


y_pred = model.predict([X_1[4000:],X_2[4000:]])
np.save('ffn_out',y_pred)
