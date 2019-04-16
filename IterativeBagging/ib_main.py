#coding=utf-8


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from IterativeBagging import IterativeBagging
from sklearn.ensemble import BaggingRegressor
import numpy as np


dataPath="./Dataset/krkopt.data"   #数据路径
headname=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'label']    #列名
data=pd.read_table(dataPath,sep=',',error_bad_lines=False,names=headname)          #读数据
#print (data.shape)          #打印出数据规模
le = preprocessing.LabelEncoder()         #数据预处理
le.fit(data['feature1'] )
data['feature1']=le.transform(data['feature1'])         #将feature1列str转为int
le.fit(data['feature3'])
data['feature3']=le.transform(data['feature3'])         #将feature3列str转为int
le.fit(data['feature5'])
data['feature5']=le.transform(data['feature5'])         #将feature5列str转为int
le.fit(data['label'])
sample_classes = le.transform(le.classes_)
data['label']=le.transform(data['label'])               #将label列str转为int
#print (data)
X = data [['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']]
y = data ['label']
#分割70%训练集和30%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


#iterativeBagging
iterativeBagging = IterativeBagging(X_train, y_train,n_estimators=500)
iterativeBagging.train()
predictions = iterativeBagging.predict(X_test)
print (explained_variance_score(y_test, predictions, multioutput='uniform_average'))

#Bagging
clf = BaggingRegressor(base_estimator=None, bootstrap=True, n_estimators=500, random_state=0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print (explained_variance_score(y_test, predictions, multioutput='uniform_average'))
