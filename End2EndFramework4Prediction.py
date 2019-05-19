from tensorflow import keras
import numpy as np



model_NN1 = keras.models.load_model('nn1_5000.h5')
# print(yyy.shape)

print("reading data")
X=np.load('nn2_inp_img1000.npy')/255


y_pred_NN1 = model_NN1.predict(X)

model_FFN = keras.models.load_model('ffn_4000.h5')
X_2=y_pred

y_pred_FFN = model_FFN.predict([X,X_2])

np.save('FinalOutput',y_pred_FFN)

# np.save('nn2_inp_mask4000',y_pred)





