# -*- coding: utf-8 -*-
import numpy as np
import commonFunc
from sklearn import svm
import matplotlib.pyplot as plt


# 写入数据
print('*'*20,"程序开始-读取数据",'*'*20)
X_Train = commonFunc.parseFile('../UCI HAR Dataset/train/X_train.txt')
Y_Train = commonFunc.parseFile('../UCI HAR Dataset/train/y_train.txt').flatten()
X_Test = commonFunc.parseFile('../UCI HAR Dataset/test/X_test.txt')
Y_Test = commonFunc.parseFile('../UCI HAR Dataset/test/y_test.txt').flatten()
print("数据读取完成~\n")

# SVM_C
def SVM_ParamC_Test(SVM_Type, CRange, SVM_DEGREE=3, SVM_COEF0=0, SVM_GAMMA='scale'):
    print('-'*20,"SVM参数惩罚因子C调节",'-'*20)
    plt.figure(figsize=(8,8),dpi=80)
    activityLabels = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
    plt.ion()
    CsetDraw = []
    FscoreDraw = []
    meanDraw = []
    
    if SVM_Type == 'linear':
        SVMParams = "Kernel(linear)"
    elif SVM_Type == 'poly':
        SVMParams = "Kernel(poly) degree({0}) coef0({1}) gamma({2})".format(SVM_DEGREE, SVM_COEF0, SVM_GAMMA)
    elif SVM_Type == 'rbf':
        SVMParams = "Kernel(rbf) gamma({0})".format(SVM_GAMMA)
    elif SVM_Type == 'sigmoid':
        SVMParams = "Kernel(sigmoid)  coef0({0}) gamma({1})".format(SVM_COEF0, SVM_GAMMA)
    else:
        print("No SVM type!")
        return
        
        
    for Cset in CRange:
        print("参数C为：{0:.2f}".format(Cset))
        plt.cla()
    
        plt.title("SVM Adjust Params-C\n {0}".format(SVMParams))
        plt.ylabel("F_Score")
        plt.xlabel("C_Value")
        plt.xlim(left=0,right=CRange[-1])
        plt.grid(True)
                
        clf = svm.SVC(kernel=SVM_Type, C=Cset, degree=SVM_DEGREE, gamma=SVM_GAMMA,coef0=SVM_COEF0, probability=False)
        clf.fit(X_Train,Y_Train)
        Y_predict = clf.predict(X_Test)
        prec, rec, f_score=commonFunc.checkAccuracy(Y_Test, Y_predict)
        
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
  
    
# SVM_Poly_Degree
def SVM_PolyDegree(SVM_DEGREE=[3], SVM_C = 0.94, SVM_COEF0=0.5, SVM_GAMMA='scale'):
    print('-'*20,"SVM多项式次数测试",'-'*20)
    plt.figure(figsize=(8,8),dpi=80)
    activityLabels = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
    plt.ion()
    DegreeSetDraw = []
    FscoreDraw = []
    meanDraw = []
    
        
    for SVM_DegreeSet in SVM_DEGREE:
        print("参数Degree为：{0}".format(SVM_DegreeSet))
        plt.cla()
    
        ParamSet = "Kernel(poly) C({0}) Gamma({1}) Coef0({2})".format(SVM_C,SVM_GAMMA,SVM_COEF0)
        plt.title("SVM Adjust Params-Degree\n{0}".format(ParamSet))
        plt.ylabel("F_Score")
        plt.xlabel("Degree_Value")
        plt.xlim(left=0,right=SVM_DEGREE[-1]+5)
        plt.grid(True)
                
        clf = svm.SVC(kernel="poly", C=SVM_C, degree=SVM_DegreeSet, gamma=SVM_GAMMA,coef0=SVM_COEF0, probability=False)
        clf.fit(X_Train,Y_Train)
        Y_predict = clf.predict(X_Test)
        prec, rec, f_score=commonFunc.checkAccuracy(Y_Test, Y_predict)
        
        FscoreDraw.append(list(f_score))
        DegreeSetDraw.append(SVM_DegreeSet)
        meanDraw.append(np.mean([f_score]))
        for modeIndex in range(6):
            plt.plot(DegreeSetDraw,[mode[modeIndex] for mode in FscoreDraw],
                     color='#'+'8'*(5-modeIndex)+'0'+'B'*(modeIndex),
                     label=activityLabels[modeIndex])
        plt.plot(DegreeSetDraw,meanDraw,'k--',label='Mean')
            
        plt.legend(loc='upper right', shadow=True)
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()
   
    
# SVM_Poly_GAMMA
def SVM_PolyGamma(SVM_DEGREE=2, SVM_C = 0.90, SVM_COEF0=0, SVM_GAMMA='scale'):
    print('-'*20,"SVM的GAMMA测试",'-'*20)
    plt.figure(figsize=(8,8),dpi=80)
    activityLabels = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
    plt.ion()
    GammaSetDraw = []
    FscoreDraw = []
    meanDraw = []
    
        
    for SVM_GammaSet in SVM_GAMMA:
        print("参数Gamma为：{0}".format(SVM_GammaSet))
        plt.cla()
    
        ParamSet = "Kernel(poly) C({0}) Degree({1}) Coef0({2})".format(SVM_C,SVM_DEGREE,SVM_COEF0)
        plt.title("SVM Adjust Params-Gamma\n{0}".format(ParamSet))
        plt.ylabel("F_Score")
        plt.xlabel("Gamma_Value")
        plt.xlim(left=0,right=1)
        plt.grid(True)
                
        clf = svm.SVC(kernel="poly", C=SVM_C, degree=SVM_DEGREE, gamma=SVM_GammaSet,coef0=SVM_COEF0, probability=False)
        clf.fit(X_Train,Y_Train)
        Y_predict = clf.predict(X_Test)
        prec, rec, f_score=commonFunc.checkAccuracy(Y_Test, Y_predict)
        
        FscoreDraw.append(list(f_score))
        GammaSetDraw.append(SVM_GammaSet)
        meanDraw.append(np.mean([f_score]))
        for modeIndex in range(6):
            plt.plot(GammaSetDraw,[mode[modeIndex] for mode in FscoreDraw],
                     color='#'+'8'*(5-modeIndex)+'0'+'B'*(modeIndex),
                     label=activityLabels[modeIndex])
        plt.plot(GammaSetDraw,meanDraw,'k--',label='Mean')
            
        plt.legend(loc='lower right', shadow=True)
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()
    
if __name__=='__main__':
    SVM_ParamC_Test('poly',np.arange(0.1,15,0.1), SVM_DEGREE=2,SVM_GAMMA='scale')
#    SVM_PolyDegree(SVM_DEGREE=range(1,15), SVM_C = 0.90, SVM_COEF0=0, SVM_GAMMA=0.01)
#    SVM_PolyGamma(SVM_GAMMA = list(np.arange(0.01,1,0.01)))
    print("程序执行完成~\n")   