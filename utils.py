import os
import cv2
import numpy as np


def saveData(dataPath,maskPath,dataSave,maskSave):
    files=os.listdir(dataPath)
    X=[]
    Y=[]
    for i in range(len(files)):
        # img_i=np.pad(cv2.imread(dataPath+files[i],1), [(0,20),(0,0),(0,0)],'constant'  )
        # print(img_i.shape)
        # X.append(img_i)#reading BGR
        img_o=np.pad(cv2.imread(maskPath+'_groundtruth_(1)_Image'+files[i],0), [(0,20),(0,0)],'constant'  )
        Y.append(img_o)#reading greyscale
        if i%100==0:
            print(i)
    # np.save(dataSave,np.asarray(X))
    np.save(maskSave,np.asarray(Y))

# saveData('./Data/','./Mask/','data_new','mask_new')
# saveData('./Mask/','mask',0)

# import tensorflow.keras as keras

def getoutput(model_name,x_name):
    print("loading model")
    model=keras.models.load_model(model_name)
    model.summary()
    print("loading data")
    x=np.load(x_name)
    x=x[5000:]
    print("getting prediction")
    y_pred = model.predict(x)
    print(x.shape)
    np.save('nn2_out_mask5000',y_pred)


getoutput('nn1.h5','mask.npy')






#everything after this is almost useless


# X=np.load('data.npy')
# Y=np.load('mask.npy')

# print(X.shape,Y.shape)
# for i in range(3000,3100):
# 	cv2.imwrite('tmp/'+str(i)+'.png',X[i])
# 	cv2.imwrite('tmp/'+str(i)+'_mask.png',Y[i])


# def ensureDimension(path):
# 	files=os.listdir(path)
# 	for i in range(len(files)):
# 		cv2.imread(path+files)


# Image_original_C1_S1_I1.tiff_2a225781-a943-483e-b3f7-10d56bf6ca1b
# _groundtruth_(1)_Image_C1_S1_I1.tiff_2a225781-a943-483e-b3f7-10d56bf6ca1b
