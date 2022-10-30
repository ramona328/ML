#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# ## Data Preparation
# 
# ### Loading the dataset

# In[2]:


os.chdir('C:\\Users\\ritvi\\Python\\ML Analytics\\Data')
#Reading csv file
covid = pd.read_csv('COVID-19BehaviorData_CAN_USA.csv')
pd.options.display.max_rows=80
#Replacing __NA__ w NA : i14_health_other column
covid_data = covid.replace('__NA__', np.nan, regex=True)
#Replacing other missing values as NA
covid_data = covid.replace(' ',np.nan)


# ### Columns with missing values

# In[3]:


covid_data.columns[covid_data.isna().any()]


# ### Feature Selection

# In[4]:


#New dataframe with only predictors and target variables
df = pd.DataFrame.copy(covid_data.loc[:,["i9_health", "i4_health","i5_health_1","i5_health_2","i5_health_3","i5_health_4","i5_health_5","i5_health_99", "i5a_health", "i8_health","i11_health", "i12_health_1","i12_health_2","i12_health_3","i12_health_4","i12_health_5", "i12_health_6", "i12_health_11", "i12_health_13", "i12_health_20", "d1_health_1", "d1_health_2", "d1_health_3","d1_health_4","d1_health_5", "d1_health_6","d1_health_7","d1_health_8","d1_health_9","d1_health_10", "weight","gender","age", "region_state", "household_size", "household_children", "employment_status"]])
df.head(3)


# #### We picked predictors that were likely to have some logical correlation with the target variable. 

# ### Target variable selection
# #### Our target is to classify 'i9_health': "Would you isolate if you start experiencing symptoms?"
# #### Being able to predict this could help gauge compliance with isolation policies based on other answers to survey.

# In[33]:


fig, ax = plt.subplots()
df['i9_health'].value_counts().plot(ax=ax, kind='bar')


# In[5]:


pd.options.display.max_columns=80
df.info()


# In[6]:


covid_data['household_size'].value_counts()


# #### 'i5a_health' and 'i8_health' have a lot of NULL values, and will be dropped from analysis.
# #### 'household_size', 'household_children' and 'region_state' have too many different classes - also dropped.

# In[7]:


#Excluding the above columns
df = covid_data.loc[:,["i9_health", "i4_health","i5_health_1","i5_health_2","i5_health_3","i5_health_4","i5_health_5","i5_health_99", "i11_health", "i12_health_1","i12_health_2","i12_health_3","i12_health_4","i12_health_5", "i12_health_6", "i12_health_11", "i12_health_13", "i12_health_20", "d1_health_1", "d1_health_2", "d1_health_3","d1_health_4","d1_health_5", "d1_health_6","d1_health_7","d1_health_8","d1_health_9","d1_health_10", "weight","gender","age","employment_status"]]


# In[8]:


#Excluding rows with all NAs
df=df.dropna()
df.head()


# In[9]:


df['i9_health'].value_counts() 


# #### 72.7% accuracy if we classify everyone as 'Yes' for likely to isolate when experiencing symptoms

# In[25]:


plt.figure(figsize=(25,8))
plt.hist(df['i11_health'])


# In[11]:


#DF with categorical predictors
X_cat = df.loc[:,["i4_health","i5_health_1","i5_health_2","i5_health_3","i5_health_4","i5_health_5","i5_health_99", "i11_health", "i12_health_1","i12_health_2","i12_health_3","i12_health_4","i12_health_5", "i12_health_6", "i12_health_11", "i12_health_13", "i12_health_20", "d1_health_1", "d1_health_2", "d1_health_3","d1_health_4","d1_health_5", "d1_health_6","d1_health_7","d1_health_8","d1_health_9","d1_health_10","gender", "employment_status"]]
#DF with numeric predictors
X_num = df.loc[:,["weight","age"]]


# In[12]:


#DF with Target
Y=df.loc[:,["i9_health"]]


# In[13]:


#Encoding all categorical variables to nominal
from sklearn.preprocessing import LabelEncoder 
X_cat=X_cat.apply(LabelEncoder().fit_transform)


# In[14]:


X_cat.head(2)


# In[15]:


#Combining predictors into one data frame
frames=[X_cat, X_num]
X=pd.concat(frames, axis=1)
X


# In[16]:


#Normalizing 'age'
age = pd.DataFrame.copy(X['age'])
norAge = (age - min(age))/(max(age)-min(age))
X['age']=norAge
X['age']


# ### Correlation heatmap of all predictors used

# In[46]:


fig, ax = plt.subplots(figsize=(25, 25))
dataplot = sns.heatmap(X.corr(), cmap="YlGnBu", annot=True, fmt='.2f')
plt.show()


# In[17]:


#Splitting into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)


# #### Accuracy

# In[18]:


from sklearn.neighbors import KNeighborsClassifier
knnc=KNeighborsClassifier(n_neighbors=17)
knnc.fit(X_train, Y_train.values.ravel())
print(f'''KNN Score for k=17: {round(knnc.score(X_test, Y_test)*100,2)}%''')


# ### We ran the KNN-Classifier for K=1 to K=50, and found that K=17 had the highest accuracy.

# #### Confusion Matrix

# In[19]:


from sklearn.metrics import plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')
plot_confusion_matrix(knnc, X_test, Y_test)

