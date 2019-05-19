
# coding: utf-8

# In[1]:


#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    


# In[2]:


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


x_train=X[:9000]
x_test=X[9000:]
y_train=Y[:9000]
y_test=Y[9000:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = y_train.astype('float32') / 255
y_test = y_test.astype('float32') / 255

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
del X
del Y


def image_generator(x_train,y_train, batch_size = 64):
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

# In[7]:


#plt.imshow(x_train[2000])


# In[8]:


#X


# In[9]:


#plt.imshow(y_train[2000])


# In[10]:


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


model=keras.models.Model(inp,o)


# In[12]:


model.summary()


# In[13]:


model.compile(loss='binary_crossentropy',
      optimizer= 'Adam' ,
      metrics=['accuracy'])


# In[14]:
print("compiled model")

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)


# In[15]:


yy=np.expand_dims(y_train, axis=3)#(y_train,(9000,300,400,1))


# In[16]:


yy.shape


# In[17]:


y_train.shape


# In[18]:
bs=4
trainGen=image_generator(x_train,y_train, batch_size = bs)
print("batch Size",bs)
H = model.fit_generator(trainGen,steps_per_epoch=int(x_train.shape[0]/bs)	,epochs=4)
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

yyy=np.expand_dims(y_test, axis=3)#(y_train,(9000,300,400,1))


# In[16]:


print(yyy.shape)
score = model.evaluate(x_test,yyy )


# In[34]:


print(score)


# In[35]:


# model.save('CorrectActivations.h5')


# In[36]:


# from sklearn.metrics import f1_score,confusion_matrix,precision_score
# import matplotlib.mlab as mlab


# In[45]:


#y_pred = model.predict(x_test)

model.save('nn1_'+str(x_train.shape[0])+'.h5')


y_pred = model.predict(x_test)
np.save('nn2_inp_mask'+str(x_train.shape[0]),y_pred)

for i in range(100):
	cv2.imwrite('output/'+str(i)+'_pred.png',y_pred[i]*255)
	cv2.imwrite('output/'+str(i)+'_inp.png',x_test[i]*255)
	cv2.imwrite('output/'+str(i)+'_gt.png',y_test[i]*255)


