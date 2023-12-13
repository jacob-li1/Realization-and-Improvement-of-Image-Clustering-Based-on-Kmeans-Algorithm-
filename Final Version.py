import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import os, glob, shutil
from sklearn import metrics
a=''
i=0
glob_dir=''
images=[]
paths=[]
target=[]
a=input("图片是否有标签： yes or no ")
if a=='yes':
    while True:
        b=input("请输入图片所在的文件夹:")
        if b=='exit':
             break
        photo = os.listdir(b)#文件路径自己改
        glob_dir=b+'/*.jpg'
        images.extend([cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)])
        paths .extend( [file for file in glob.glob(glob_dir)])
        for name in photo:
            target.append(i)
        i += 1
elif a=='no':
        b=input("请输入图片所在的文件夹:")
        glob_dir=b+'/*.jpg'
        images.extend([cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)])
        paths .extend( [file for file in glob.glob(glob_dir)])
images = np.array(np.float32(images).reshape(len(images), -1)/255)
model = tf.keras.applications.MobileNetV2(include_top=False,weights='imagenet', input_shape=(224, 224, 3))
predictions = model.predict(images.reshape(-1, 224, 224, 3))
pred_images = predictions.reshape(images.shape[0], -1)
d=input("请选择聚类的方式：1：Kmeans++  2：MiniBatchKmeans ")
print("正在分析类的个数K......\n")
sil = []
kl = []
kmax = 10
for k in range(2, kmax+1):
    kmeans2 = KMeans(n_clusters = k).fit(pred_images)
    labels = kmeans2.labels_
    sil.append(silhouette_score(pred_images, labels, metric = 'euclidean'))
    kl.append(k)
    plt.figure(1)
    plt.plot(kl, sil)
plt.ylabel('Silhoutte Score')
plt.xlabel('K')
inertia=[]
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k,random_state=0).fit(pred_images)
    inertia.append(np.sqrt(kmeans.inertia_))
    plt.figure(2)
plt.plot(range(2,kmax+1),inertia,'o-')
plt.ylabel('SSE')
plt.xlabel('K')
CH = []
for k in range(2,kmax+1):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(pred_images)
    CH.append(davies_bouldin_score(pred_images,estimator.labels_))
plt.figure(3)
plt.xlabel('k')
plt.ylabel('davies_bouldin_score')
plt.plot(range(2,kmax+1),CH,'o-')
CH=[]
for i in range(2,kmax+1):
	#构建并训练模型
	kmeans = KMeans(n_clusters = i,random_state=123).fit(pred_images)
	CH.append(calinski_harabasz_score(pred_images,kmeans.labels_))
X = range(2,kmax+1)
plt.figure(4)
plt.xlabel('k')
plt.ylabel('calinski_harabasz__score')
plt.plot(X,CH,'o-')
plt.show()
k=int(input("请输入类的个数K:"))
c=input("请输入聚类结果存放的文件夹:")
print("开始聚类......")
if(d=='1'):
    startTime=time.time()
    kmodel =  KMeans(n_clusters = k,init='k-means++', n_init=400,max_iter=50000,tol=0.0000001,verbose=0,random_state=None,copy_x=True,algorithm='auto')
    kmodel.fit(pred_images)
    kpredictions = kmodel.predict(pred_images)
    shutil.rmtree(c)
    for i in range(k):
       os.makedirs(c+"\cluster" + str(i))
    for i in range(len(paths)):
       shutil.copy2(paths[i], c+"\cluster"+str(kpredictions[i]))
    if(a=='yes'):
        score = metrics.adjusted_rand_score(target, kpredictions)
        print("聚类结果评价指标兰德系数为："+str(score)+"\n")
    print("聚类完毕")
    print("运行时间："+str(time.time()-startTime)+"\n")
elif(d=='2'):
    startTime=time.time()
    kmodel=MiniBatchKMeans(n_clusters=k,init='k-means++',n_init=400,max_iter=50000,tol=0.0000001,batch_size=100, random_state=None) 
    kmodel.fit(pred_images)
    kpredictions = kmodel.predict(pred_images)
    shutil.rmtree(c)
    for i in range(k):
       os.makedirs(c+"\cluster" + str(i))
    for i in range(len(paths)):
       shutil.copy2(paths[i], c+"\cluster"+str(kpredictions[i]))
    if(a=='yes'):
        score = metrics.adjusted_rand_score(target, kpredictions)
        print("聚类结果评价指标兰德系数为："+str(score)+"\n")
    print("聚类完毕")
    print("运行时间："+str(time.time()-startTime)+"\n")