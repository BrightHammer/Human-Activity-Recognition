# %% import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

# 绘制混淆矩阵


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.rainbow):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列,labels
    - normalize : True:显示百分比, False:显示个数
    """
    cm = np.array(cm, dtype=np.int32)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black")

    axes = plt.gca()
    axes.set_xlim([-0.5, len(classes)-0.5])
    axes.set_ylim([-0.5, len(classes)-0.5])

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()


# ----------------debug---
""" cf_matrix = []
for i in range(0, 6):
    tmp = []
    for j in range(0, 6):
        tmp.append(i+j)
    cf_matrix.append(tmp)

print(cf_matrix)
LABELS = ["STANDING", "SITTING", "LAYING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]
plot_confusion_matrix(cf_matrix,LABELS,normalize=True) """

# %% --------------------箱型图/小提琴图----------
src_data = pd.read_csv("../UCI HAR Dataset/test.csv")
#src_data.head()
subject_data = src_data[src_data["subject"]==2]
subject_data = subject_data.drop(columns=['subject'])

# %% 
LABELS = ["STANDING", "SITTING", "LAYING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]
mean_val=pd.DataFrame(columns=[],index=[],data=[])
for i in range(3):
    tmp_mean=subject_data[(subject_data["Activity"]==LABELS[i])].mean()
    tmp_mean.head()
    mean_val[LABELS[i]]=np.array(tmp_mean).T



#sns.violinplot(x = "subject", y = "tBodyAcc-mean()-X", data = src_data,hue = "Activity")


# %%
