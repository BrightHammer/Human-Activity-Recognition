# %% import package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
# plt.rcParams['font.family'] = 'Dejavu Sans'
plt.rcParams['font.sans-serif'] = ['SimHei'] #正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
# %% 读取features.txt获得特征数量
features = list()
with open('../UCI HAR dataset/features.txt') as f:
    features = [line.split()[1] for line in f.readlines()]
print('No of Features: {}'.format(len(features)))

# %% 训练集，从txt读取数据，转为pandas的DataFrame
X_train = pd.read_csv('../UCI HAR dataset/train/X_train.txt', delim_whitespace=True, header=None)
X_train.columns = features

# 加入subject列
X_train['subject'] = pd.read_csv('../UCI HAR dataset/train/subject_train.txt', header=None, squeeze=True)

y_train = pd.read_csv('../UCI HAR dataset/train/y_train.txt', names=['Activity'], squeeze=True)
y_train_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING'})

# 合并到同一个DataFrame
train = X_train
train['Activity'] = y_train
train['ActivityName'] = y_train_labels
train.sample()
train.shape

# %% 测试集，从txt读取数据，转为pandas的DataFrame
X_test = pd.read_csv('../UCI HAR dataset/test/X_test.txt', delim_whitespace=True, header=None)
X_test.columns = features

# 加入subject列
X_test['subject'] = pd.read_csv('../UCI HAR dataset/test/subject_test.txt', header=None, squeeze=True)

y_test = pd.read_csv('../UCI HAR dataset/test/y_test.txt', names=['Activity'], squeeze=True)
y_test_labels = y_test.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING'})

# 合并到同一个DataFrame
test = X_test
test['Activity'] = y_test
test['ActivityName'] = y_test_labels
test.sample()
test.shape

# %% Data Cleaning
# 检查重复项
print('No of duplicates in train: {}'.format(sum(train.duplicated())))
print('No of duplicates in test : {}'.format(sum(test.duplicated())))

# 检查 NaN/null 项
print('We have {} NaN/Null values in train'.format(train.isnull().values.sum()))
print('We have {} NaN/Null values in test'.format(test.isnull().values.sum()))

# 检查均匀度
plt.figure(figsize=(16,8))
plt.title('Data provided by each user', fontsize=20)
sns.countplot(x='subject',hue='ActivityName', data = train)
plt.savefig('figure/1.png')
plt.show()

plt.title('No of Datapoints per Activity', fontsize=15)
sns.countplot(train.ActivityName)
plt.xticks(rotation=90)
plt.show()

# 修改特征名称
columns = train.columns

columns = columns.str.replace('[()]','')
columns = columns.str.replace('[-]', '')
columns = columns.str.replace('[,]','')

train.columns = columns
test.columns = columns

test.columns

# %% 保存为csv，方便其他地方调用
# train.to_csv('../UCI HAR Dataset/csv_files/train.csv', index=False)
# test.to_csv('../UCI HAR Dataset/csv_files/test.csv', index=False)

# %% Exploratory Data Analysis

sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(train, hue='ActivityName', size=6,aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMagmean', hist=False).add_legend()
plt.annotate(u"静态活动", xy=(-0.956,17), xytext=(-0.9, 23), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate(u"动态活动", xy=(0,3), xytext=(0.2, 9), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.title(u'各类活动平均加速度大小数据分布', fontsize=20)
plt.tight_layout()
plt.savefig('figure/2.png')
plt.show()

# %% 
# for plotting purposes taking datapoints of each activity to a different dataframe
df1 = train[train['Activity']==1]
df2 = train[train['Activity']==2]
df3 = train[train['Activity']==3]
df4 = train[train['Activity']==4]
df5 = train[train['Activity']==5]
df6 = train[train['Activity']==6]

plt.figure(figsize=(14,7))
plt.subplot(2,2,1)
plt.title('静态活动')
sns.distplot(df4['tBodyAccMagmean'],color = 'r',hist = False, label = 'Sitting')
sns.distplot(df5['tBodyAccMagmean'],color = 'm',hist = False,label = 'Standing')
sns.distplot(df6['tBodyAccMagmean'],color = 'c',hist = False, label = 'Laying')
plt.axis([-1.01, -0.5, 0, 35])
plt.legend(loc='center')

plt.subplot(2,2,2)
plt.title('动态活动')
sns.distplot(df1['tBodyAccMagmean'],color = 'red',hist = False, label = 'Walking')
sns.distplot(df2['tBodyAccMagmean'],color = 'blue',hist = False,label = 'Walking Up')
sns.distplot(df3['tBodyAccMagmean'],color = 'green',hist = False, label = 'Walking down')
plt.legend(loc='center right')


plt.tight_layout()
plt.show()

# %% 加速度大小均值——箱型图
#plt.figure(figsize=(7,7))
plt.title(u'各类活动加速度大小数据箱型图', fontsize=15)
sns.boxplot(x='ActivityName', y='tBodyAccMagmean',data=train, showfliers=False, saturation=1)
plt.ylabel('Acceleration Magnitude mean')
plt.axhline(y=-0.7, xmin=0.1, xmax=0.9,dashes=(5,5), c='g')
plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
plt.xticks(rotation=40)
plt.show()

# %% 重力加速度分量方向与 X 轴夹角
sns.boxplot(x='ActivityName', y='angleXgravityMean', data=train)
plt.axhline(y=0.08, xmin=0.1, xmax=0.9,c='m',dashes=(5,3))
plt.title('重力加速度方向与X轴夹角', fontsize=15)
plt.xticks(rotation = 40)
plt.show()

# %% 重力加速度分量方向与 Y 轴夹角
sns.boxplot(x='ActivityName', y='angleYgravityMean', data = train, showfliers=False)
plt.title('重力加速度方向与Y轴夹角', fontsize=15)
plt.xticks(rotation = 40)
plt.axhline(y=-0.22, xmin=0.1, xmax=0.8, dashes=(5,3), c='m')
plt.show()

# %% PCA 降维与可视化
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  #调整n_component参数以设置降维后的维度
data_pca = pca.fit_transform(train.drop(['subject', 'Activity','ActivityName'], axis=1))
print(type(data_pca))
print(data_pca)

df = pd.DataFrame({'x':data_pca[:,0], 'y':data_pca[:,1] ,'label':train['ActivityName']})
        
sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=8,
            palette="Set1",markers=['^','v','s','o', '1','2'])
plt.title('PCA数据降维与可视化', fontsize=15)
img_name = "figure/PCA.png"
plt.tight_layout()
plt.savefig(img_name)
plt.show()

# %% t-sne 分析
from sklearn.manifold import TSNE
# performs t-sne with different perplexity values and their repective plots..

def perform_tsne(X_data, y_data, perplexities, n_iter=1000, img_name_prefix='t-sne'):
        
    for index,perplexity in enumerate(perplexities):
        # perform t-sne
        print('\nperforming tsne with perplexity {} and with {} iterations at max'.format(perplexity, n_iter))
        X_reduced = TSNE(verbose=2, perplexity=perplexity).fit_transform(X_data)
        print('Done..')
        
        # prepare the data for seaborn         
        print('Creating plot for this t-sne visualization..')
        df = pd.DataFrame({'x':X_reduced[:,0], 'y':X_reduced[:,1] ,'label':y_data})
        
        # draw the plot in appropriate place in the grid
        sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=8,\
                   palette="Set1",markers=['^','v','s','o', '1','2'])
        # plt.title("perplexity : {} and max_iter : {}".format(perplexity, n_iter))
        plt.title('t-SNE数据降维与可视化（混乱度=50）', fontsize=15)
        # img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplexity, n_iter)
        print('saving this plot as image in present working directory...')
        # plt.tight_layout()
        plt.savefig('figure/t-sne')
        plt.show()
        print('Done')

X_pre_tsne = train.drop(['subject', 'Activity','ActivityName'], axis=1)
y_pre_tsne = train['ActivityName']
perform_tsne(X_data = X_pre_tsne,y_data=y_pre_tsne, perplexities =[50])

# %%
