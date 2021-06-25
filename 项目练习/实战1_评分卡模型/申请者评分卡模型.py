#!/usr/bin/env python
# coding: utf-8

# ## 导入所需要的库

# In[569]:


# 数据分析和整理
import pandas as pd
import numpy as np
import random as rnd

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 特征工程
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 机器学习模型
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#模型检验
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc


# ## 获取数据

# 

# In[184]:


#读取用户申请信息表数据
train_data=pd.read_csv("cs-training.csv",index_col=0)


# ## 数据初步探索

# In[185]:


train_data.shape


# In[186]:


train_data.head()


# In[187]:


train_data.tail()


# In[188]:


train_data.info()


# In[189]:


#查看缺失值
train_data.isnull().mean()


# In[190]:


#查看数据分布
train_data.describe([0.01,0.1,0.25,0.5,0.75,0.9,0.99]).T


# ## 数据初步分析

# In[191]:


# 样本分布情况
train_data['SeriousDlqin2yrs'].value_counts().plot.bar()


# ### 填充缺失值

# In[192]:


def fill_missing(data,to_fill):
    df=data.copy()
    columns=[*df.columns]
    columns.remove(to_fill)
    columns.remove("NumberOfDependents")
    x=df.loc[:,columns]
    y=df.loc[:,to_fill]
    x_train=x.loc[df[to_fill].notnull()]
    x_pred=x.loc[df[to_fill].isnull()]
    y_train=y.loc[df[to_fill].notnull()]
    model=RandomForestRegressor(random_state=0,
                               n_estimators=200,
                               max_depth=3,
                               n_jobs=-1)
    model.fit(x_train,y_train)
    pred=model.predict(x_pred)
    df.loc[df[to_fill].isnull(),to_fill]=pred
    return df


# In[193]:


train_data=fill_missing(train_data,'MonthlyIncome')


# In[194]:


train_data.info()


# In[195]:


train_data[['SeriousDlqin2yrs', 'NumberOfDependents']].groupby(['NumberOfDependents']).mean().sort_values(by='SeriousDlqin2yrs', ascending=False)


# In[196]:


# 家庭成员数不同违约情况不同，说明NumberOfDependents对违约产生影响
# 有部分缺失值


# In[197]:


train_data.dropna(inplace=True)


# ### 处理异常值

# In[198]:


train_data.info()


# In[199]:


col=["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse","NumberOfTimes90DaysLate"]


# In[200]:


train_data[col].plot.box(vert=False)


# In[201]:


for i in col:
    train_data=train_data.loc[train_data[i]<90]


# In[202]:


train_data[col].plot.box(vert=False)


# ### 划分数据集

# In[203]:


Y=train_data['SeriousDlqin2yrs']
X=train_data.iloc[:,1:]


# In[204]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# In[205]:


model_data=pd.concat([Y_train,X_train],axis=1)


# In[206]:


test_data=pd.concat([Y_test,X_test],axis=1)


# In[ ]:





# ### 数据初步统计分析

# In[207]:


model_data.age.plot.hist(bins=30)


# In[208]:


model_data.describe([0.1,0.99]).T


# In[209]:


model_data.MonthlyIncome.loc[model_data.MonthlyIncome].plot.hist(bins=50)


# In[210]:


model_data.MonthlyIncome.loc[model_data.MonthlyIncome<23400].plot.hist(bins=50)


# In[219]:


model_data.MonthlyIncome.min()


# In[211]:


model_data.NumberOfOpenCreditLinesAndLoans.plot.hist(bins=50)


# In[ ]:





# In[212]:


#绘制变量关系的相关关系矩阵
model_corr = model_data.corr()
model_corr


# In[213]:


#画出相关性热力图
a = plt.subplots(figsize=(15,9))#调整画布大小
a = sns.heatmap(model_corr, vmin=-1, vmax=1 , annot=True , square=True)#画热力图


# ### 变量选择

# #### 分组

# ##### 类别较多的分组

# In[214]:


# 自动最优分组函数


# In[251]:


def auto_bin(DF,X,Y,n=3,iv=True,woe=False,q=20):
    DF=DF[[X,Y]].copy()
    #按照等深进行分箱
    DF['qcut'],bins=pd.qcut(DF[X],q=20,retbins=True,duplicates='drop')
    #统计每个分箱中SeriousDlqin2yrs 0,1的数量
    count_y0=DF.loc[DF[Y]==0].groupby(by='qcut')[Y].count()
    count_y1=DF.loc[DF[Y]==1].groupby(by='qcut')[Y].count()
    #每个区间的上界，下届，0的频数，1的频数
    num_bins=[*zip(bins,bins[1:],count_y0,count_y1)]

    #确保每一个分组中都包含0和1
    for i in range(20):
        #如果第一组没有0或1，向后合并
        if 0 in num_bins[0][2:]:
            num_bins[0:2]=[(
                num_bins[0][0],
                num_bins[1][1],
            num_bins[0][2]+num_bins[1][2],
            num_bins[0][3]+num_bins[1][3])]
            continue
        #其他组没有0或1，向前合并
        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i-1:i+1]=[(
                num_bins[i-1][0],
                num_bins[i][1],
                num_bins[i-1][2]+num_bins[i][2],
                num_bins[i-1][3]+num_bins[i-1][3])]
                break
        else:
                break
    #定义计算woe的函数
    def get_woe(num_bins):
        columns=["min","max","good","bad"]
        df=pd.DataFrame(num_bins,columns=columns)
        df['total']=df.good+df.bad
        df['percebtage']=df.total/df.total.sum()
        df["woe"]=np.log((df.good/df.good.sum())/(df.bad/df.bad.sum()))
        return df
    #定义计算IV值函数
    def get_iv(bins_df):
        rate=((bins_df.good/bins_df.good.sum())-(bins_df.bad/bins_df.bad.sum()))
        iv=np.sum(rate*bins_df.woe)
        return iv
#基于卡方检验进行分析。
#原假设：观察频数与期望的频数没有差异
#如果P很小意味着两者偏离程度比较大，应该拒绝原假设
#获取两两之间的卡方检验的置信度，如果p比较大
#n为保留的最小组数
    
    while len(num_bins)>n:
        pvs=[]
        for i in range(len(num_bins)-1):
            x1=num_bins[i][2:]
            x2=num_bins[i+1][2:]
            pv=scipy.stats.chi2_contingency([x1,x2])[1]
            pvs.append(pv)
        #将P值最大两组进行合并  
        i=pvs.index(max(pvs))
        num_bins[i:i+2]=[(
        num_bins[i][0],
        num_bins[i+1][1],
        num_bins[i][2]+num_bins[i+1][2],
        num_bins[i][3]+num_bins[i+1][3])]
        #打印每组的分箱信息（woe，iv）
        bins_df=get_woe(num_bins)
        if iv:
            print(f"{X}分{len(num_bins):2}组的IV值",get_iv(bins_df))
        if woe:
            print(bins_df)
    return get_woe(num_bins)


# In[239]:


# #查看每一个字段，观察是否需要自动分箱
for i in model_data.columns:
    print (i)
    model_counts = model_data[i].value_counts().count()
    print (model_counts)


# 根据最优分箱函数确认RevolvingUtilizationOfUnsecuredLines，age，DebtRatio，MonthlyIncome的最优分箱数
# - RevolvingUtilizationOfUnsecuredLines  5组比较合适
# - age       6组比较合适
# - DebtRatio   5组比较合适
# - MonthlyIncome   3组比较合适
# - NumberOfOpenCreditLinesAndLoans  5组比较合适

# In[241]:


#根据最优分箱函数确认每个字段的最优分箱数
auto_bin(model_data,'RevolvingUtilizationOfUnsecuredLines','SeriousDlqin2yrs',n=3,woe=True,q=20)


# In[243]:


#根据最优分箱函数确认age字段的最优分箱数
auto_bin(model_data,'age','SeriousDlqin2yrs',n=3,woe=True,q=20)


# In[244]:


#根据最优分箱函数确认DebtRatio的最优分箱数
auto_bin(model_data,'DebtRatio','SeriousDlqin2yrs',n=3,woe=True,q=20)


# In[245]:


#根据最优分箱函数确认MonthlyIncome的最优分箱数
auto_bin(model_data,'MonthlyIncome','SeriousDlqin2yrs',n=3,woe=True,q=20)


# In[246]:


#根据最优分箱函数确认NumberOfOpenCreditLinesAndLoans的最优分箱数
auto_bin(model_data,'NumberOfOpenCreditLinesAndLoans','SeriousDlqin2yrs',n=3,woe=True,q=20)


# In[ ]:





# In[247]:


#手动添加自动分箱数
auto_col_bins={'RevolvingUtilizationOfUnsecuredLines':5,
               'age':6,
               'DebtRatio':5,
               'MonthlyIncome':3,
               'NumberOfOpenCreditLinesAndLoans':5
}


# In[248]:


for col in auto_col_bins:
    print(col)


# In[249]:


for col in auto_col_bins:
    print(auto_col_bins[col])


# In[256]:


for col in auto_col_bins:
    bins_df=auto_bin(model_data,col,'SeriousDlqin2yrs',n=auto_col_bins[col],iv=False,q=20)
    print(bins_df)


# In[409]:


#保存分箱数据
bins_of_col={}
#生成自动分箱的分箱区间和分箱后的IV值
for col in auto_col_bins:
    bins_df=auto_bin(model_data,col,'SeriousDlqin2yrs',n=auto_col_bins[col],iv=False,q=20)
    
    #保证区间最小值为-np.inf,,最大值为np.inf
    bins_list=sorted(set(bins_df["min"]).union(bins_df["max"]))
    bins_list[0],bins_list[-1]=-np.inf,np.inf
    bins_of_col[col]=bins_list


# In[389]:


bins_of_col


# ##### 类别较少的手动分组

# In[305]:


model_data['NumberRealEstateLoansOrLines'].value_counts()


# In[407]:


# 类别较少的手动分组
hand_bins={'NumberOfTime30-59DaysPastDueNotWorse':[0,1,2,13],
          'NumberOfTimes90DaysLate':[0,1,2,17],
          'NumberOfTime60-89DaysPastDueNotWorse':[0,1,2,9],
          'NumberOfDependents':[0,1,2,3,10],
          'NumberRealEstateLoansOrLines':[0,1,2,54]}
hand_bins={k:[-np.inf,*v[:-1],np.inf] for k,v in hand_bins.items()}


# ##### 合并两种分箱数据

# In[410]:


#合并分箱数据
bins_of_col.update(hand_bins)
bins_of_col


# In[ ]:





# In[411]:


#计算分箱数据的IV值
def get_iv(df,col,y,bins):
    df=df[[col,y]].copy()
    df['cut']=pd.cut(df[col],bins)
    bins_df=df.groupby('cut')[y].value_counts().unstack()
    bins_df['woe']=np.log((bins_df[0]/bins_df[0].sum())/(bins_df[1]/bins_df[1].sum()))
    iv=np.sum((bins_df[0]/bins_df[0].sum()-bins_df[1]/bins_df[1].sum())*bins_df.woe)
    return iv,bins_df


# In[460]:


#保存IV值信息
iv_values={}
#保存woe信息
woe_values={}
for col in bins_of_col:
    iv_woe=get_iv(model_data,col,'SeriousDlqin2yrs',bins_of_col[col])
    iv_values[col],woe_values[col]=iv_woe#保存IV值信息


# In[461]:


keys,values=zip(*iv_values.items())
nums=range(len(keys))
plt.barh(nums,values)
plt.yticks(nums,keys)
for i ,v in enumerate(values):
    plt.text(v,i-.2,f"{v:3f}")


# In[ ]:





# #### WOE转换

# In[462]:


model_data.head()


# In[463]:


woe_values


# In[464]:


#创建一个空的DataFrame
model_woe=pd.DataFrame(index=model_data.index)


# In[417]:


for col in bins_of_col:
    model_woe[col]=pd.cut(model_data[col],bins_of_col[col]).map(woe_values[col]['woe'])


# In[418]:


model_woe['SeriousDlqin2yrs']=model_data["SeriousDlqin2yrs"]


# In[419]:


model_woe.head()


# In[ ]:





# ### 构建模型

# In[ ]:


#因变量
Y=model_woe.SeriousDlqin2yrs


# In[438]:


#自变量
X=model_woe.drop(['SeriousDlqin2yrs','NumberOfDependents'],axis=1)


# In[547]:


test_woe=pd.DataFrame(index=test_data.index)


# In[548]:


for col in bins_of_col:
    test_woe[col]=pd.cut(test_data[col],bins_of_col[col]).map(woe_values[col]['woe'])


# In[549]:


test_woe['SeriousDlqin2yrs']=test_data['SeriousDlqin2yrs']


# In[550]:


test_Y=test_woe['SeriousDlqin2yrs']


# In[551]:


test_X=test_woe.drop(['SeriousDlqin2yrs','NumberOfDependents'],axis=1)


# #### 调参
# 

# In[553]:


lr = LogisticRegression()

param = {'C':[0.001,0.01,0.1,1,10],"max_iter":[100,250]}

clf = GridSearchCV(lr,param,cv=3,n_jobs=-1,verbose=1,scoring="roc_auc")

clf.fit(X,Y)


# In[554]:


clf.cv_results_ 


# In[555]:


clf.best_params_


# In[573]:


#将最佳参数传入训练模型
lr = LogisticRegression(C=0.1, max_iter=100)


# In[574]:


lr.fit(X,Y)


# In[575]:


#预测结果
lr.predict(test_X)


# In[576]:


predictions_pro = lr.predict_proba(test_X)
predictions_pro


# In[559]:


print(metrics.confusion_matrix(test_Y, test_proba))


# In[561]:


print(metrics.classification_report(test_Y, test_proba))


# #### roc曲线

# In[580]:


#图可以显示中文
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False


# In[585]:


false_positive_rate, recall, thresholds = roc_curve(test_Y,predictions_pro[:,1])
roc_auc = auc(false_positive_rate, recall)
plt.title("ROC Curves")
plt.plot(false_positive_rate, recall, 'r', label='AUC = % 0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('假阳性率')
plt.ylabel('召回率')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




