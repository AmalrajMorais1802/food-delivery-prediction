#!/usr/bin/env python
# coding: utf-8

# In[89]:


import os
import pandas as pd
import numpy as np
os.chdir("C:\\Users\\antony.morais\\Desktop\\Amalraj\\Food Delivery Prediction\\Participants Data")
os.getcwd()


# In[90]:


train=pd.read_excel('Data_Train.xlsx')
test=pd.read_excel('Data_Test.xlsx')


# In[91]:


train.head(5)


# In[92]:


train.count()


# In[93]:


#To Check if Null Values are Present
train.isnull().sum()


# In[94]:


train['res_id'] = train.Restaurant.str[3:]
del train['Restaurant']


# In[95]:


for i in range(0,len(train)) :
    train['Location'][i]=train['Location'][i].split(',')[-1].strip()
for i in range(0,len(test)) :
    test['Location'][i]=test['Location'][i].split(',')[-1].strip()


# In[96]:


train.groupby(['Location']).size()


# In[97]:


train['deltime']=train['Delivery_Time'].str[:-8]
del train['Delivery_Time']


# In[98]:


train['deltime']=train['deltime'].astype(int)


# In[99]:


train.Average_Cost = train.Average_Cost.str[1:]
train.Minimum_Order = train.Minimum_Order.str[1:]
test.Average_Cost = test.Average_Cost.str[1:]
test.Minimum_Order = test.Minimum_Order.str[1:]


# In[100]:


train['Rating']=train['Rating'].replace(to_replace ="-", value ="0")
train['Rating']=train['Rating'].replace(to_replace ="Temporarily Closed", value ="0")
train['Rating']=train['Rating'].replace(to_replace ="Opening Soon", value ="0")
test['Rating']=test['Rating'].replace(to_replace ="-", value ="0")
test['Rating']=test['Rating'].replace(to_replace ="Temporarily Closed", value ="0")
test['Rating']=test['Rating'].replace(to_replace ="Opening Soon", value ="0")
test['Rating']=test['Rating'].replace(to_replace ="NEW", value="0")
train['Rating']=train['Rating'].replace(to_replace ="NEW", value="0")


# In[101]:


train[["Rating"]] = train[["Rating"]].apply(pd.to_numeric)
def mod_rat(cols):
    Rating = cols[0]
    if Rating==5:
        return round(train[(train["Rating"]!=5) & (train["Rating"]!=0)]['Rating'].mean(),1)
    else:
        return Rating
train["Rating"] = train[["Rating"]].apply(mod_rat,axis=1)


# In[102]:


test[["Rating"]] = test[["Rating"]].apply(pd.to_numeric)
def mod_rat_test(cols):
    Rating = cols[0]
    if Rating==5:
        return round(test[(test["Rating"]!=5) & (test["Rating"]!=0)]['Rating'].mean(),1)
    else:
        return Rating
test["Rating"] = test[["Rating"]].apply(mod_rat_test,axis=1)


# In[103]:


train.groupby(['Rating']).size()


# In[104]:


train=train[train['Average_Cost']!='or']
train['Average_Cost']=train['Average_Cost'].str.replace(",","").astype(int)
test=test[test['Average_Cost']!='or']
test['Average_Cost']=test['Average_Cost'].str.replace(",","").astype(int)


# In[105]:


train.groupby(['Average_Cost']).size()


# In[106]:


#Univariate Plot
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)
sns.distplot(train['Average_Cost'])


# In[107]:


plt.boxplot(train['Average_Cost'])
plt.plot()
#pd.DataFrame(train['Average_Cost']).T.boxplot(vert=False)
#plt.show()


# In[108]:


train['Votes']=train['Votes'].replace(to_replace ="-", value ="0")
train['Reviews']=train['Reviews'].replace(to_replace ="-", value ="0")
train['Votes']=train['Votes'].astype(int)
train['Reviews']=train['Reviews'].astype(int)
test['Votes']=test['Votes'].replace(to_replace ="-", value ="0")
test['Reviews']=test['Reviews'].replace(to_replace ="-", value ="0")
test['Votes']=test['Votes'].astype(int)
test['Reviews']=test['Reviews'].astype(int)


# In[109]:


#train.groupby(['Reviews']).size()
#761
#train['Votes'].nunique()
#1103


# In[110]:


train.groupby(['deltime']).size()


# In[111]:


train["Minimum_Order"] = train["Minimum_Order"].astype('int')
test["Minimum_Order"] = test["Minimum_Order"].astype('int')


# In[112]:


train.dtypes


# In[113]:


#train['Cuisines'].nunique() #2177
train['Cuisines'].head(100)


# In[114]:


shuf_loc=[]
for i in train['Location'] :
    shuf_loc.append(i.split(' ')[0].strip())
train['Location']=shuf_loc
shuf_loc=[]
for i in test['Location'] :
    shuf_loc.append(i.split(' ')[0].strip())
test['Location']=shuf_loc


# In[115]:


def mod_loc(cols):
    Location = cols[0]
    if Location=='Gurgoan':
        return 'Delhi'
    elif (Location=='Electronic'or Location=='Marathalli' or Location=='Majestic' or Location=='Whitefield'):
        return 'Bangalore'
    elif Location=='Begumpet':
        return 'Hyderabad'
    elif Location=='Maharashtra':
        return 'Mumbai'
    elif Location=='Timarpur':
        return 'Delhi'
    elif Location=='Gurgaon':
        return 'Delhi'
    elif Location=='Noida':
        return 'Delhi'
    else :
        return Location
train['Location']= train[['Location']].apply(mod_loc,axis=1)
test['Location']= test[['Location']].apply(mod_loc,axis=1)


# In[116]:


train.groupby(['Location']).size()


# In[117]:


train.head()


# In[118]:


#!pip install pyod
max_iqr=train.Average_Cost.quantile(0.95)
max_iqr=int(max_iqr)
#print(max_iqr)
def mod_cst(cols):
    cst = cols[0]
    if cst>max_iqr:
        return max_iqr
    else:
        return cst
train["Average_Cost"] = train[["Average_Cost"]].apply(mod_cst,axis=1)


# In[119]:


max_order=train.Minimum_Order.quantile(0.95)
max_iqr=int(max_order)
train["Minimum_Order"] = train[["Minimum_Order"]].apply(mod_cst,axis=1)


# In[120]:


max_votes=train.Votes.quantile(0.95)
max_iqr=int(max_votes)
train["Votes"] = train[["Votes"]].apply(mod_cst,axis=1)
max_Reviews=train.Reviews.quantile(0.95)
max_iqr=int(max_Reviews)
train["Reviews"] = train[["Reviews"]].apply(mod_cst,axis=1)


# In[121]:


# Checking for Variance Distribution
train_df=train.copy()
del train_df['Cuisines']
del train_df['Location']
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.7 * (1 - .7)))
high_var=sel.fit_transform(train_df)
#Tells the no of columns worth out of the total number of columns
#print(len(train_df.columns[sel.get_support()]))
useless = [column for column in train_df.columns
                    if column not in train_df.columns[sel.get_support()]]
print(useless)


# In[122]:


os.getcwd()


# In[123]:


train.sort_values(['res_id', 'Rating'], ascending=[True, False],inplace=True)
train.reset_index(drop=True,inplace=True)
dup_chk=pd.DataFrame(train.duplicated(subset=['Average_Cost','Minimum_Order','Rating','Votes','Reviews','res_id'], keep='first'))
dup_chk.columns=["Dup"]
dup_chk.groupby(['Dup']).size()
train_fin=train[dup_chk['Dup']==False]
train_fin=train_fin.sample(frac=1,random_state=4)
train_fin.reset_index(drop=True,inplace=True)
train_fin.reset_index(drop=True).to_csv("train_final.csv",index=False)


# In[124]:


#############  ONE HOT ENCODING   ######################

one_hot = pd.get_dummies(train_fin['Location'])
train_fin = train_fin.join(one_hot)
del train_fin['Location']
one_hot = pd.get_dummies(test['Location'])
test = test.join(one_hot)
del test['Location']
train_fin["Bangalore"] = train_fin["Bangalore"].astype('category')
train_fin["Delhi"] = train_fin["Delhi"].astype('category')
train_fin["India"] = train_fin["India"].astype('category')
train_fin["Kolkata"] = train_fin["Kolkata"].astype('category')
train_fin["Hyderabad"] = train_fin["Hyderabad"].astype('category')
train_fin["Mumbai"] = train_fin["Mumbai"].astype('category')
train_fin["Pune"] = train_fin["Pune"].astype('category')
test["Bangalore"] = test["Bangalore"].astype('category')
test["Delhi"] = test["Delhi"].astype('category')
test["Mumbai"] = test["Mumbai"].astype('category')
test["India"] = test["India"].astype('category')
test["Kolkata"] = test["Kolkata"].astype('category')
test["Hyderabad"] = test["Hyderabad"].astype('category')
test["Pune"] = test["Pune"].astype('category')


# In[125]:


bins = [-0.01, 25, 150, 10000]
labels = ['Less_Rv','Med_Rv','Highly_Rv']
train_fin['rev_binned'] = pd.cut(train_fin['Reviews'], bins=bins,labels=labels)
del train_fin['Reviews']
test['rev_binned'] = pd.cut(test['Reviews'], bins=bins,labels=labels)
del test['Reviews']


# In[126]:


one_hot = pd.get_dummies(train_fin['rev_binned'])
train_fin = train_fin.join(one_hot)
del train_fin['rev_binned']
one_hot = pd.get_dummies(test['rev_binned'])
test = test.join(one_hot)
del test['rev_binned']


# In[127]:


train_fin["Less_Rv"] = train_fin["Less_Rv"].astype('category')
train_fin["Med_Rv"] = train_fin["Med_Rv"].astype('category')
train_fin["Highly_Rv"] = train_fin["Highly_Rv"].astype('category')
test["Less_Rv"] = test["Less_Rv"].astype('category')
test["Med_Rv"] = test["Med_Rv"].astype('category')
test["Highly_Rv"] = test["Highly_Rv"].astype('category')


# In[128]:


train_fin.head()


# In[129]:


#Do Standardization for Votes, Rating, Minimum_Order, Average_Cost
# from sklearn import preprocessing
# standardized_votes = preprocessing.scale(pd.DataFrame(train_fin['Votes']))
# train_fin['standardized_votes']=standardized_votes
# standardized_rating = preprocessing.scale(pd.DataFrame(train_fin['Rating']))
# train_fin['standardized_rating']=standardized_rating
# standardized_minord = preprocessing.scale(pd.DataFrame(train_fin['Minimum_Order']))
# train_fin['standardized_minord']=standardized_minord
# standardized_avgcst = preprocessing.scale(pd.DataFrame(train_fin['Average_Cost']))
# train_fin['standardized_avgcst']=standardized_avgcst
# del train_fin['Votes']
# del train_fin['Minimum_Order']
# del train_fin['Rating']
# del train_fin['Average_Cost']


# In[130]:


votes_min=train_fin['Votes'].min()
votes_max=train_fin['Votes'].max()
for i in range(0,len(train_fin)) :
    train_fin.loc[i,'Votes']=round((train_fin.loc[i,'Votes']-votes_min)/(votes_max-votes_min),2)
rating_min=train_fin['Rating'].min()
rating_max=train_fin['Rating'].max()
for i in range(0,len(train_fin)) :
    train_fin.loc[i,'Rating']=round((train_fin['Rating'][i]-rating_min)/(rating_max-rating_min),2)
minord_min=train_fin['Minimum_Order'].min()
minord_max=train_fin['Minimum_Order'].max()
for i in range(0,len(train_fin)) :
    train_fin.loc[i,'Minimum_Order']=round((train_fin['Minimum_Order'][i]-minord_min)/(minord_max-minord_min),2)
avgcost_min=train_fin['Average_Cost'].min()
avgcost_max=train_fin['Average_Cost'].max()
for i in range(0,len(train_fin)) :
    train_fin.loc[i,'Average_Cost']=round((train_fin['Average_Cost'][i]-avgcost_min)/(avgcost_max-avgcost_min),2)


# In[131]:


votes_min=test['Votes'].min()
votes_max=test['Votes'].max()
for i in range(0,len(test)) :
    test.loc[i,'Votes']=round((test.loc[i,'Votes']-votes_min)/(votes_max-votes_min),2)
rating_min=test['Rating'].min()
rating_max=test['Rating'].max()
for i in range(0,len(test)) :
    test.loc[i,'Rating']=round((test['Rating'][i]-rating_min)/(rating_max-rating_min),2)
minord_min=test['Minimum_Order'].min()
minord_max=test['Minimum_Order'].max()
for i in range(0,len(test)) :
    test.loc[i,'Minimum_Order']=round((test['Minimum_Order'][i]-minord_min)/(minord_max-minord_min),2)
avgcost_min=test['Average_Cost'].min()
avgcost_max=test['Average_Cost'].max()
for i in range(0,len(test)) :
    test.loc[i,'Average_Cost']=round((test['Average_Cost'][i]-avgcost_min)/(avgcost_max-avgcost_min),2)


# In[132]:


# bins = [-0.01, 0, 50, 1000]
# labels = ['Order=0','1<=Order<=50','Order>50']
# train_fin['order_binned'] = pd.cut(train_fin['Minimum_Order'], bins=bins,labels=labels)
# train_fin['order_binned'].value_counts()
# one_hot = pd.get_dummies(train_fin['order_binned'])
# train_fin = train_fin.join(one_hot)
# del train_fin['order_binned']
# train_fin["Order=0"] = train_fin["Order=0"].astype('category')
# train_fin["1<=Order<=50"] = train_fin["1<=Order<=50"].astype('category')
# train_fin["Order>50"] = train_fin["Order>50"].astype('category')
# del train_fin['Minimum_Order']


# In[133]:


# bins = [-0.01, 0, 50, 1000]
# labels = ['Order=0','1<=Order<=50','Order>50']
# test['order_binned'] = pd.cut(test['Minimum_Order'], bins=bins,labels=labels)
# test['order_binned'].value_counts()
# one_hot = pd.get_dummies(test['order_binned'])
# test = test.join(one_hot)
# del test['order_binned']
# test["Order=0"] = test["Order=0"].astype('category')
# test["1<=Order<=50"] = test["1<=Order<=50"].astype('category')
# test["Order>50"] = test["Order>50"].astype('category')
# del test['Minimum_Order']


# In[134]:


del train_fin['Cuisines']
del train_fin['res_id']


# In[135]:


del test['Cuisines']
del test['Restaurant']


# In[136]:


###########################    CORRELATION    ##############################


# In[155]:


traincor=train_fin.copy()


# In[156]:


# traincor.dtypes


# In[157]:


traincor.head()


# In[158]:


del traincor['Highly_Rv']
del traincor['Less_Rv']


# In[159]:


traincor.head()


# In[160]:


traincor["Bangalore"] = traincor["Bangalore"].astype('int32')
traincor["Delhi"] = traincor["Delhi"].astype('int32')
traincor["Mumbai"] = traincor["Mumbai"].astype('int32')
traincor["India"] = traincor["India"].astype('int32')
traincor["Hyderabad"] = traincor["Hyderabad"].astype('int32')
traincor["Pune"] = traincor["Pune"].astype('int32')
traincor["Kolkata"] = traincor["Kolkata"].astype('int32')
#traincor["Less_Rv"] = traincor["Less_Rv"].astype('int32')
traincor["Med_Rv"] = traincor["Med_Rv"].astype('int32')
#traincor["Highly_Rv"] = traincor["Highly_Rv"].astype('int32')
traincor["Minimum_Order"] = traincor["Minimum_Order"].astype('int32')


# In[161]:


co_op=traincor.corr(method ='pearson')


# In[162]:


co_op['index']=co_op.index
co_op.index


# In[163]:


co_op_mlt=co_op.melt(id_vars =['index'], value_vars =['Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'deltime',
       'Bangalore', 'Delhi', 'Hyderabad', 'India', 'Kolkata', 'Mumbai', 'Pune','Med_Rv'] ,
           var_name ='Variable_column', value_name ='Value_column') 


# In[164]:


co_op_mlt['Abs_value']=abs(co_op_mlt['Value_column'])
co_op_mlt['eliminate'] = np.where((co_op_mlt['Variable_column'] == co_op_mlt['index']) , 1, 0)
co_op_mlt_r=co_op_mlt[co_op_mlt.eliminate==0]
del co_op_mlt_r['eliminate']


# In[165]:


co_op_mlt_r.sort_values(by=['Abs_value'],ascending=False)


# In[166]:


del train_fin['Highly_Rv']
del train_fin['Less_Rv']
del test['Highly_Rv']
del test['Less_Rv']


# In[167]:


train_fin.head()


# In[168]:


test.head()


# In[169]:


train_fin.sort_values(['Average_Cost','Rating','Votes'], ascending=[True, True, True],inplace=True)
train_fin.reset_index(drop=True,inplace=True)
dup_chk=pd.DataFrame(train_fin.duplicated(subset=None, keep='first'))
dup_chk.columns=["Dup"]
dup_chk.groupby(['Dup']).size()
train_final=train_fin[dup_chk['Dup']==False]
train_final=train_final.sample(frac=1,random_state=4)
train_final.reset_index(drop=True,inplace=True)
train_final.reset_index(drop=True).to_csv("train_final_mod.csv",index=False)


# In[170]:


train_final.shape


# Modeling

# In[171]:


#!pip install joblib
import joblib


# In[172]:


train_final.dtypes


# In[173]:


test_final=test.copy()
train_final['Minimum_Order']=train_final['Minimum_Order'].astype('float32')
train_final['Average_Cost']=train_final['Average_Cost'].astype('float32')
train_final['Votes']=train_final['Votes'].astype('float32')
train_final['Rating']=train_final['Rating'].astype('float32')
train_final['Bangalore']=train_final['Bangalore'].astype('int32')
train_final['Delhi']=train_final['Delhi'].astype('int32')
train_final['Mumbai']=train_final['Mumbai'].astype('int32')
train_final['Pune']=train_final['Pune'].astype('int32')
train_final['India']=train_final['India'].astype('int32')
train_final['Kolkata']=train_final['India'].astype('int32')
train_final['Hyderabad']=train_final['Kolkata'].astype('int32')
test_final['Minimum_Order']=test_final['Minimum_Order'].astype('float32')
test_final['Average_Cost']=test_final['Average_Cost'].astype('float32')
test_final['Votes']=test_final['Votes'].astype('float32')
test_final['Rating']=test_final['Rating'].astype('float32')
test_final['Bangalore']=test_final['Bangalore'].astype('int32')
test_final['Delhi']=test_final['Delhi'].astype('int32')
test_final['Mumbai']=test_final['Mumbai'].astype('int32')
test_final['Pune']=test_final['Pune'].astype('int32')
test_final['India']=test_final['India'].astype('int32')
test_final['Kolkata']=test_final['India'].astype('int32')
test_final['Hyderabad']=test_final['Kolkata'].astype('int32')


# In[177]:


#=########################    AUTOML    ########################
from automl import automl
import random
random.seed(1000)
X_train, X_test, y_train, y_test = automl.load_data(df=train_final, dv_name='deltime', model='Regression', metric='r2', ignore_col_name=[],trainsize=0.7)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
import os
os.chdir(r'C:\Users\antony.morais\Desktop\Amalraj\Food Delivery Prediction\Participants Data\automl_op')


# In[179]:


#For a population of 5, we see the best model
tpot_obj = automl.perform_model_optimization(X_train,y_train,cv=7,gen=5,pop=11,config=None, int_result_folder="AutoML_Pipelines_Output")
automl.prepare_output_file(X_train, y_train,X_test, y_test,tpot_obj,10)


# In[180]:


import statsmodels.api as sm


# In[181]:


X = train_final.copy()
del X['deltime']
y = train_final['deltime']


# In[182]:


#Divide target variable y 100
y=y/60
#y


# In[74]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[75]:


# Note the difference in argument order
#model = sm.OLS(y, X.astype(float)).fit()
model = sm.OLS(y_train, X_train.astype(int)).fit()
y_pred = model.predict(X_test) # make the predictions by the model
#y_pred = model.predict(test_final)


# In[79]:


# Print out the statistics
#model.summary()


# In[83]:


# from sklearn.linear_model import LinearRegression
# lin_model = LinearRegression()
# lin_model.fit(X_train, y_train)
# from sklearn.metrics import r2_score
# # model evaluation for training set
# y_train_predict = lin_model.predict(X_train)
# rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
# r2 = r2_score(y_train, y_train_predict)
# print("The model performance for training set")
# print("--------------------------------------")
# print('RMSE is {}'.format(rmse))
# print('R2 score is {}'.format(r2))
# print("\n")
# # model evaluation for testing set
# y_test_predict = lin_model.predict(X_test)
# rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
# r2 = r2_score(y_test, y_test_predict)
# print("The model performance for testing set")
# print("--------------------------------------")
# print('RMSE is {}'.format(rmse))
# print('R2 score is {}'.format(r2))


# In[77]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))
rms


# In[422]:


y_pred1=y_pred*60
ypa=[]
for i in y_pred1:
    ypa.append(round(i,0))
def myround(x, base=5):
    return base * round(x/base)
ypaf=[]
for i in ypa:
    ypaf.append(myround(i))
y_pred2=pd.DataFrame(ypaf)
y_pred2.columns=["Delivery_Time"]


# In[424]:


for i in range(0,len(y_pred2['Delivery_Time'])):
    y_pred2['Delivery_Time'][i]=str(y_pred2['Delivery_Time'][i])+' minutes'


# In[425]:


y_pred2.reset_index(drop=True).to_csv("submission_linreg.csv", sep=',', encoding='utf-8',index=False)


# XGBOOST

# In[426]:


import xgboost as xgb


# In[427]:


X.head()


# In[430]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[431]:


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, 
                          alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)


# In[432]:


X_train.dtypes


# In[433]:


y_pred = xg_reg.predict(X_test)


# In[434]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))


# In[435]:


y_pred1=y_pred*100
ypa=[]
for i in y_pred1:
    ypa.append(round(i,0))
def myround(x, base=5):
    return base * round(x/base)
ypaf=[]
for i in ypa:
    ypaf.append(myround(i))
y_pred2=pd.DataFrame(ypaf)
y_pred2.columns=["Delivery_Time"]
for i in range(0,len(y_pred2['Delivery_Time'])):
    y_pred2['Delivery_Time'][i]=str(y_pred2['Delivery_Time'][i])+' minutes'


# In[436]:


y_pred2.reset_index(drop=True).to_csv("submission_xgboost.csv", sep=',', encoding='utf-8',index=False)


# In[ ]:




