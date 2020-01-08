# -*- coding: utf-8 -*-
import numpy as np
import commonFunc
from sklearn.feature_selection import RFECV
from sklearn import svm
import matplotlib.pyplot as plt
from time import perf_counter
import os
from visualization import plot_confusion_matrix


def main():
    # 写入数据
    print('*'*20,"程序开始-读取数据",'*'*20)
    X_Train = commonFunc.parseFile('../UCI HAR Dataset/train/X_train.txt')
    Y_Train = commonFunc.parseFile('../UCI HAR Dataset/train/y_train.txt').flatten()
    X_Test = commonFunc.parseFile('../UCI HAR Dataset/test/X_test.txt')
    Y_Test = commonFunc.parseFile('../UCI HAR Dataset/test/y_test.txt').flatten()
    activityLabels = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
    print("数据读取完成~\n")
    
    # 参数设置
    print('*'*20,"设置参数表",'*'*20)
    KERNELSET = 'linear'
    DEGREE = 0.91
    CSET = 3
    COEF0 = 0
    GAMMA = 'scale'
    print("SVM:")
    print("kernel:{0} \t C:{1}".format(KERNELSET,CSET))
    print("RFE:")
    STEPSET = 5
    MINFEATURETOSET = 300
    CROSSVALIDATION = 20
    CPUCHANNEL = 6
    print("estimate:SVM \t step:{0} \t min_feature_to_select:{1} \t CrossValidation:{2} \t CPUChannel:{3} \n"
          .format(STEPSET,MINFEATURETOSET,CROSSVALIDATION,CPUCHANNEL))
    
    # 特征选择
    print('*'*20,"读取特征文件",'*'*20)
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
        estimator = svm.SVC(kernel=KERNELSET, degree=DEGREE, C=CSET, coef0=COEF0, gamma=GAMMA,  probability=False)
        selector = RFECV(estimator, step=STEPSET, min_features_to_select = MINFEATURETOSET, cv=CROSSVALIDATION, n_jobs = CPUCHANNEL)
        selector = selector.fit(X_Train, Y_Train)
        mask = selector.get_support()
        print("特征选择完成!")
        print("用时 {0:.2f}mins".format((perf_counter()-start)/60))
        print("特征选择数量： {0}".format(sum(mask==1)))
        np.savetxt(maskSaveName,mask,fmt='%d')
        
    # 画图
    plt.figure(figsize=(14,14))
    plt.subplot(2,2,(1,2))
    plt.imshow(mask.reshape(1,-1),cmap='tab20c_r')
    plt.title("Feature Selected: {0}".format(sum(mask==1)),fontsize=14,y=2.5)
    plt.ylim([-5,5])
    plt.xlabel("Feature Index(Deeper Color means Selected)",fontsize=10)
#    plt.show()
    print('\n')

    
     # 选择特征抽取
    print('*'*20,"特征选择后的数据结果",'*'*20)
    X_Train_selected = X_Train[:,mask]
    X_Test_selected = X_Test[:,mask]
    clf_selected = svm.SVC(kernel=KERNELSET, degree=DEGREE, C=CSET, coef0=COEF0, gamma=GAMMA, probability=False)
    clf_selected.fit(X_Train_selected, Y_Train)
    Y_predict_selected = clf_selected.predict(X_Test_selected)
    prec_selected, rec_selected, f_score_selected=commonFunc.checkAccuracy(Y_Test, Y_predict_selected)
    print("训练结果：")
    print("准确率：{0}\n召回率：{1}\nF1度量：{2}".format(prec_selected, rec_selected, f_score_selected))
    
     # 混淆矩阵
    plt.subplot(2,2,3)
    cm = commonFunc.createConfusionMatrix(Y_predict_selected, Y_Test)
    plot_confusion_matrix(cm, activityLabels, normalize=False, title='Selected_F Confusion matrix')
    print('\n')
    
    # 原始数据的训练结果
    print('*'*20,"特征选择前的数据结果",'*'*20)
    clf = svm.SVC(kernel=KERNELSET, degree=DEGREE, C=CSET, coef0=COEF0, gamma=GAMMA, probability=False)
    clf.fit(X_Train,Y_Train)
    Y_predict = clf.predict(X_Test)
    prec, rec, f_score=commonFunc.checkAccuracy(Y_Test, Y_predict)
    print("训练结果：")
    print("准确率：{0}\n召回率：{1}\nF1度量：{2}".format(prec, rec, f_score))
    
    # 混淆矩阵
    plt.subplot(2,2,4)
    cm = commonFunc.createConfusionMatrix(Y_predict, Y_Test)
    plot_confusion_matrix(cm, activityLabels, normalize=False, title='All_F Confusion matrix')
    
#    plt.tight_layout()
    plt.show()

        
    
if __name__=='__main__':
    main()