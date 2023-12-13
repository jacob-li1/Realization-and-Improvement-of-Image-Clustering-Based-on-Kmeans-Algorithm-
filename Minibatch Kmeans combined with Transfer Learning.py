import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
#from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil
input_dir = 'D:\\buildings'
glob_dir = input_dir + '/*.jpg'
images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
paths = [file for file in glob.glob(glob_dir)]
images = np.array(np.float32(images).reshape(len(images), -1)/255)
model = tf.keras.applications.MobileNetV2(include_top=False,weights='imagenet', input_shape=(224, 224, 3))
predictions = model.predict(images.reshape(-1, 224, 224, 3))
pred_images = predictions.reshape(images.shape[0], -1)
k = 3
kmodel=MiniBatchKMeans(n_clusters=k,init='k-means++',n_init=400,max_iter=50000,tol=0.0000001,batch_size=100, random_state=None) 
#kmodel =  KMeans(n_clusters = k,init='random', n_init=400,max_iter=50000,tol=0.0000001,verbose=0,random_state=None,copy_x=True,algorithm='auto')
kmodel.fit(pred_images)
kpredictions = kmodel.predict(pred_images)
shutil.rmtree('D:\\test1')
for i in range(k):
    os.makedirs("D:\\test1\cluster" + str(i))
for i in range(len(paths)):
    shutil.copy2(paths[i], "D:\\test1\cluster"+str(kpredictions[i]))