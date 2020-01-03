# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:55:16 2020

@author: MelonEater
"""

from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 准确率
import common
## 记载样本数据集
#iris = load_iris()
#X,y = iris.data,iris.target
## 数据集分割
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123457)
X_train=common.parseFile('../UCI HAR Dataset/train/X_train.txt')			
y_train=(common.parseFile('../UCI HAR Dataset/train/y_train.txt')).flatten() - 1		
X_test=common.parseFile('../UCI HAR Dataset/test/X_test.txt')				
y_test=(common.parseFile('../UCI HAR Dataset/test/y_test.txt')).flatten() - 1	
# 算法参数
params = {
    'booster':'gbtree',
    'objective':'multi:softmax',
    'num_class':6,
    'gamma':0.1,
    'max_depth':6,
    'lambda':2,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'slient':1,
    'eta':0.1,
    'seed':1000,
    'nthread':4,
}
 
plst = params.items()
 
# 生成数据集格式
dtrain = xgb.DMatrix(X_train,y_train)
num_rounds = 5000
# xgboost模型训练
model = xgb.train(plst,dtrain,num_rounds)
 
# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)
 
# 计算准确率
accuracy = accuracy_score(y_test,y_pred)
print('accuarcy:%.2f%%'%(accuracy*100))
 
# 显示重要特征
plot_importance(model)
plt.show()
model.save_model('test10000.model')
model.dump_model('dump.raw.txt','featmap.txt')