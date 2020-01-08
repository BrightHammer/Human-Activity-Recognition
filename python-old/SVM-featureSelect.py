# -*- coding: utf-8 -*-
"""
@author: MelonEater
"""

import numpy as np
import common
from sklearn.feature_selection import RFECV
from sklearn import svm
import matplotlib.pyplot as plt
from time import perf_counter
import os

# 写入数据
print('*'*20,"程序开始-读取数据",'*'*20)
X_Train = common.parseFile('../UCI HAR Dataset/train/X_train.txt')
Y_Train = common.parseFile('../UCI HAR Dataset/train/y_train.txt').flatten()
X_Test = common.parseFile('../UCI HAR Dataset/test/X_test.txt')
Y_Test = common.parseFile('../UCI HAR Dataset/test/y_test.txt').flatten()

def main():
    # 特征选择
    maskSaveName = "SVM-features-mask.out"
    if(os.path.exists(maskSaveName)):
        print("存在特征文件，开始读取...")
        maskInteger = np.loadtxt(maskSaveName)
        mask = (maskInteger==1)
        print("读取完成，准备显示...")
        print("特征选择数量： {0}".format(sum(mask==1)))
    else:
        print("特征文件不存在~")
        print("开始特征选择...")
        start = perf_counter()
        estimator = svm.SVC(kernel='linear', C=0.9, probability=False)
        selector = RFECV(estimator, step=5, min_features_to_select = 300, cv=20, n_jobs = 6)
        selector = selector.fit(X_Train, Y_Train)
        mask = selector.get_support()
        print("特征选择完成!")
        print("用时 {0:.2f}mins".format((perf_counter()-start)/60))
        print("特征选择数量： {0}".format(sum(mask==1)))
        np.savetxt(maskSaveName,mask,fmt='%d')
        
    # 画图
    plt.matshow(mask.reshape(1,-1),cmap='tab20c_r')
    plt.title("Feature Selected: {0}".format(sum(mask==1)),fontsize=14,y=2.5)
    plt.ylim([-5,5])
    plt.xlabel("Feature Index(Deeper Color means Selected)",fontsize=10)
    plt.show()
    print('\n')


    
     # 选择特征抽取
    print('*'*20,"特征选择后的数据结果",'*'*20)
    X_Train_selected = X_Train[:,mask]
    X_Test_selected = X_Test[:,mask]
    clf_selected = svm.SVC(kernel='linear', C=0.9, probability=False)
    clf_selected.fit(X_Train_selected, Y_Train)
    Y_predict_selected = clf_selected.predict(X_Test_selected)
    prec_selected, rec_selected, f_score_selected=common.checkAccuracy(Y_Test, Y_predict_selected)
    print("训练结果：")
    print("准确率：{0}\n召回率：{1}\nF1度量：{2}".format(prec_selected, rec_selected, f_score_selected))
    print("混淆矩阵：")
    print(common.createConfusionMatrix(Y_predict_selected, Y_Test))
    print('\n')
    
    # 原始数据的训练结果
    print('*'*20,"特征选择前的数据结果",'*'*20)
    clf = svm.SVC(kernel='linear', C=0.9, probability=False)
    clf.fit(X_Train,Y_Train)
    Y_predict = clf.predict(X_Test)
    prec, rec, f_score=common.checkAccuracy(Y_Test, Y_predict)
    print("训练结果：")
    print("准确率：{0}\n召回率：{1}\nF1度量：{2}".format(prec, rec, f_score))
    print("混淆矩阵：")
    print(common.createConfusionMatrix(Y_predict, Y_Test))


def drawParamTest():
    print('*'*20,"绘图开始-读取数据",'*'*20)
    plt.figure(figsize=(8,8),dpi=80)
    activityLabels = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
    plt.ion()
    CsetDraw = []
    FscoreDraw = []
    meanDraw = []
    
    
    for Cset in np.arange(0.01,1.01,0.01):
        print("参数C为：{0:.2f}".format(Cset))
        plt.cla()
        
        plt.title("SVM Adjust Params-C")
        plt.grid(True)
                
        clf = svm.SVC(kernel='linear', C=Cset, probability=False)
        clf.fit(X_Train,Y_Train)
        Y_predict = clf.predict(X_Test)
        prec, rec, f_score=common.checkAccuracy(Y_Test, Y_predict)
        
        FscoreDraw.append(list(f_score))
        CsetDraw.append(Cset)
        meanDraw.append(np.mean([f_score]))
        for modeIndex in range(6):
            plt.plot(CsetDraw,[mode[modeIndex] for mode in FscoreDraw],
                     color='#'+'8'*(5-modeIndex)+'0'+'B'*(modeIndex),
                     label=activityLabels[modeIndex])
        plt.plot(CsetDraw,meanDraw,'k--',label='Mean')
            
        plt.legend(loc='upper right', shadow=True)
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()
        
        
    
if __name__=='__main__':
    main()
#    drawParamTest()