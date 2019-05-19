from tensorflow import keras
import numpy as np



model = keras.models.load_model('ffn_4000.h5')
# print(yyy.shape)

print("reading data")

print("reading data")
X=np.load('nn2_inp_img5000.npy')/255
Y=np.load('nn2_out_mask5000.npy')/255

# X = X[:4000]
# Y = Y[:4000]


score = model.evaluate(X,np.expand_dims(Y[4000:], axis=3) )


# In[34]:


print(score)


# model.save('ffn_'+str(x_train.shape[0])+'.h5')


# y_pred = model.predict([X_1[4000:],X_2[4000:]])
# np.save('ffn_out',y_pred)


