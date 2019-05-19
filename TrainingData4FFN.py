from tensorflow import keras
import numpy as np



model = keras.models.load_model('nn1_5000.h5')
# print(yyy.shape)

print("reading data")
X=np.load('nn2_inp_img5000.npy')/255
Y=np.load('nn2_out_mask5000.npy')/255

X = X[:4000]
Y = Y[:4000]

y_pred = model.predict(X)
np.save('nn2_inp_mask4000',y_pred)


# In[34]:


# print(score)