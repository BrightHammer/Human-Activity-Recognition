# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict

# 从文件中读取数据
def parseFile( file_name ):
	f = open(file_name)
	featureArray = []
	lines = f.readlines()
	for line in lines:
		feature_length = len(line.split(" "))

		raw_feature = line.split(" ")

		feature = []
        
        # 空格作为str是不能append到里面去的，所以这里空格不影响结果
		for index in range( feature_length ):
			try:
				feature.append( float( raw_feature[index] ))
			except:
				continue
		
		featureArray.append( feature )

	return np.asarray( featureArray )

# 计算衡量参数
def checkAccuracy( original , predicted , labels=[1,2,3,4,5,6] ):
	TP = defaultdict(list)
	FP = defaultdict(list)
	FN = defaultdict(list)

	precision = []
	recall = []
	f_score = []
	
	for i in range(len(original)):
		
		if original[i] == predicted[i]:
			TP[str(int(original[i]))].append(1)			 

		elif original[i] != predicted[i]:
			FP[str(int(predicted[i]))].append(1)			 
			FN[str(int(original[i]))].append(1)			 

	
	for label in labels:
		if len(TP[str(label)]) == 0:
			p = 0
			precision.append( p )
			r = 0
			recall.append( r  )
			fs = 0			
			f_score.append(fs)
			continue
        
		p = float( len(TP[str(label)]) ) / ( len(TP[str(label)]) + len(FP[str(label)]))
		precision.append( p )
    
		r = float( len(TP[str(label)]) ) / ( len(TP[str(label)]) + len(FN[str(label)]))
		recall.append( r  )


        # F1度量
		fs = float( 2*p*r ) / (p+r)				
		f_score.append( fs)

	return precision , recall , f_score


# 把给定的labels转为二值问题
def convertLabel(labels, posLabels, Neglabels):
	dynamic = []

	for label in labels:

		if label in posLabels:
			dynamic.append( 1 )
			 
		elif label in Neglabels:
			dynamic.append( 0 )
		else:
			print("Unknown Label: Good Gawd :)")
	return np.asarray(dynamic)


# 得到子数据序列,把RequiredLabels里面的数据都拼在一起
def getDataSubset(inputData, inputLabels, RequiredLabels): 
	subData=[]
	subLabels=[]
	for loopVar in range(len(inputLabels)):
		if inputLabels[loopVar] in RequiredLabels:
			subData.append(inputData[loopVar])
			subLabels.append(inputLabels[loopVar])
	return np.asarray(subData), np.asarray(subLabels)

# 创建混淆矩阵
def createConfusionMatrix(predictedYLabels,originalYLabels,labelList=[1,2,3,4,5,6]):
    confusionMatrix = np.zeros((len(labelList),len(labelList)))

    if len(originalYLabels) != len(predictedYLabels):
        print('Error')
        return

    for i in range(len(originalYLabels)):
        if (predictedYLabels[i] not in labelList) or (originalYLabels[i] not in labelList):
            print('Error')
            return
        else:
            confusionMatrix[labelList.index(originalYLabels[i]),labelList.index(predictedYLabels[i])] += 1
    return confusionMatrix


# 增广特征数据时使用
# x_features = [[1,2],    k = [1,2]  则输出为[[ 1,  2,  1,  4],
#               [3,4]]                       [ 3,  4,  9, 16]]
def getPowerK( X_features, k):

	X_features_new = []

	for x_feature in X_features:
		x_feature_new = []

		for power in k:
			for x_feature_dimension in x_feature:

				x_feature_new.append( np.power(x_feature_dimension,power) )

		X_features_new.append( np.asarray(x_feature_new) )				


	return np.asarray(X_features_new)
	


# 获得加速度样本
def getAccFeatures( X_train, features_file = '../UCI HAR Dataset/features.txt'):
    f = open(features_file)
    lines = f.readlines()
    AccFeaturesList = []
    i = 0
    for line in lines:
        if not 'Gyro' in line : AccFeaturesList.append(i)
        i = i + 1
    f.close()

    features = []
    for index in AccFeaturesList:
        features.append( X_train[:,index])

    return np.transpose(np.asarray(features))


# 获得角速度样本
def getGyroFeatures( X_train,feature_file='../UCI HAR Dataset/features.txt'):
    f = open(feature_file)
    lines = f.readlines()
    GyroFeaturesList = []
    i = 0
    for line in lines:
        if not 'Acc' in line : GyroFeaturesList.append(i)
        i = i + 1
    f.close()

    features = []
    for index in GyroFeaturesList:
        features.append( X_train[:,index])
    return np.transpose(np.asarray(features))


# 根据实验人物进行数据筛选
def getSubjectData(inputXData,inputYData,requiredSubjects,subjectData = None):
    requiredSubjectDataIndexList = []
    if subjectData is None:
        subjectData = parseFile('../UCI HAR Dataset/train/subject_train.txt')
    
    for i in range(len(subjectData)):
        if int(subjectData[i]) in requiredSubjects:
            requiredSubjectDataIndexList.append(i);
    return inputXData[requiredSubjectDataIndexList,:], inputYData[requiredSubjectDataIndexList],subjectData[requiredSubjectDataIndexList]
