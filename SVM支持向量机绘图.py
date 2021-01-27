# -*- coding:utf-8 -*-
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

path = './iris2.csv'
data = np.loadtxt(path, dtype=float, delimiter=',')
x, y = np.split(data, indices_or_sections=(2,), axis=1)
x = x[:, 0:2]
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)
classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')
classifier.fit(train_data, train_label.ravel())
# print("训练集：", classifier.score(train_data, train_label))
# print("测试集：", classifier.score(test_data, test_label))
# print('train_decision_function:', classifier.decision_function(train_data))  # (90,3)
# print('predict_result:', classifier.predict(train_data))
font1 = {'family': 'Times New Roman',
         'size': 10}
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
grid_hat = classifier.predict(grid_test)
grid_hat = grid_hat.reshape(x1.shape)
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=cm_dark)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, edgecolors='k', zorder=2, cmap=cm_dark)
plt.xlabel('Positive', font1)
plt.ylabel('Negative', font1)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Hotel_review_data_two_categories', font1)
plt.show()
