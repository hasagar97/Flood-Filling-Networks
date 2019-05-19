from tensorflow import keras
import numpy as np



model = keras.models.load_model('nn1_9000.h5')
# print(yyy.shape)

print("reading data")
X_1=np.load('nn2_inp_img5000.npy')/255
X_2=np.load('nn2_inp_mask5000.npy')
Y=np.load('nn2_out_mask5000.npy')/255


score = model.evaluate([X_1[4000:],X_2[4000:]],np.expand_dims(Y[4000:], axis=3) )


# In[34]:


print(score)


# model.save('ffn_'+str(x_train.shape[0])+'.h5')


# y_pred = model.predict([X_1[4000:],X_2[4000:]])
# np.save('ffn_out',y_pred)


