#This programme using SVM based on sklearn to classify different EMG signal in 5 types movement

from sklearn.datasets import load_svmlight_file
#X_train, y_train = load_svmlight_file('heart_scale')
from sklearn import svm
import scipy.io as scio
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


Xdata = scio.loadmat('/ ')['xdata'] #the data format is mat in this programme

ydata = np.empty([3020000])        #生成y lable数据，且符合纬度
for i in range(0, 5):
    ydata[604000*i:604000*i+604000] = i


#num = Xdata.shape[0]
#num_test = num/2
#model = SVR(cache_size=7000)

RANDOM_STATE = 50
X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.50, random_state=RANDOM_STATE)

'''
classf = svm.SVC(C=1.0, kernel="rbf", degree=3, verbose=True, gamma=0.7, coef0=0.0, shrinking=True, tol=1e-3, max_iter=10)
#regr = svm.SVR(kernel="rbf", degree=3, gamma="scale", coef0=0.0, C=1.0, epsilon=0.2, tol=1e-3, shrinking=True, max_iter=1000)
classf.fit(X_train, y_train)
xpre = classf.predict(X_train)
#ypre = classf.predict((y_train).reshape(1,-1))
#print(classf.predict(X_train))

acc = accuracy_score(X_train, xpre)
print("acc")
print(acc)
'''


#     print(X_new)


'''

C=1.0
models = (
    svm.SVC(kernel="Linear", C=C),
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel="rbf", gamma=0.7, C=C)
)
models = (clf.fit(Xdata, ydata) for clf in models)
'''


#可以用，调参即可
clf = svm.SVC(gamma=0.00001, kernel='rbf', C=1.0,  verbose=True, shrinking=0, max_iter=5, decision_function_shape='ovo')
clf.fit(X_train, y_train)
pred_Xtest = clf.predict(X_test)
#pred_ytest = clf.predict((y_test).reshape(1,-1))

print("Xtest Accuracy:")
print(f"{clf.score(X_train, pred_Xtest):.2%}\n")


#print(clf.score(y_train, pred_ytest))
'''
print(clf.score(X_train, y_train))  # 精度
print(clf.score(X_test, y_test))

'''

#print(f"{accuracy_score(y_test, pred_test):.2%}\n")

#acc_rbf = sum(pred_test==y_test)/num_test
#print('accuracy is',acc_rbf)

#optimization finished, #iter = 118
#C = 0.479830
#obj = 9.722436, rho = -0.224096
#nSV = 145, nBSV = 125
#Total nSV = 145

#SVC=(C=1, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.07, kernel='rbf', max_iter=-1, probability=False, shrinking=True, tol=0.001, verbose=Truepred = clf.predict(X_train))


#可视化
x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
x2_min, x2_max = X_train[:, 1].min()-1, X_train[:, 1].max() + 1
# 获得绘图边界，这里没有区分训练数据或测试数据，根据实际需求选择即可
h = (x1_max - x1_min) / 100
# h为采样点间隔，可以自己设定

xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
# 由meshgrid函数生成对应区域内所有点的横纵坐标，xx、yy均为尺寸为(M, N)的二维矩阵，分别对应区域内所有点的横坐标和所有点的纵坐标，同时也是区域内所有样本的第一维特征和第二维特征
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# 由训练好的SVM预测区域内所有样本的结果。由于xx、yy尺寸均为(M,N)，通过.ravel拉平并通过.c_组合，尺寸变为（M*N, 2），相当于M*N个具有两维特征的样本，输出z尺寸为(M*N,)
z = z.reshape(xx.shape)
# 将输出尺寸也转变为(M, N)以和横纵坐标对应绘制等高线图
plt.contourf(xx, yy, z, cmap=plt.cm.ocean, alpha=0.6)
# 绘制等高线图


plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])
# 标记数据中各样本
plt.title('Visualization of SVM with RBF kernel')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
