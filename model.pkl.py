b#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


transfusion_path="../input/blood-donation-new/transfusion_latest.csv"


# In[9]:


transfusion_data=pd.read_csv(transfusion_path)


# In[4]:


transfusion_data.head()


# In[10]:


transfusion_data.tail()


# In[11]:


transfusion_data.shape


# In[12]:


transfusion_data.describe()


# In[13]:


transfusion_data.info()


# In[14]:


transfusion_data.columns


# In[15]:


transfusion_data.isnull().sum()


# In[16]:



#relationship Analysis
corelation=transfusion_data.corr()
sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)


# In[17]:


sns.pairplot(transfusion_data)


# In[19]:


sns.distplot(a=transfusion_data['Frequency (times)'],kde=False,bins=70)


# In[20]:


sns.catplot(x='Frequency (times)',kind='box',data=transfusion_data)


# In[21]:


transfusion_data.rename(
    columns = {'whether he/she donated blood in March 2007': 'target'},
    inplace = True
)

transfusion_data.head(2)


# In[22]:



print('Target incidence proportions:\n')
print(round(transfusion_data.target.value_counts(normalize = True) * 100,3))


# In[23]:



#Splitting transfusion into train and test datasets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    transfusion_data.drop(columns='target'),
    transfusion_data.target,
    test_size=0.25,
    random_state=42,
    stratify=transfusion_data.target
)

print('First two rows of X_train: ')
X_train.head(4)


# In[24]:


#Selecting model using TPOT

from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(f'{idx}. {transform}')


# In[25]:


X_train.var().round(3)


# In[26]:


import numpy as np

# Copy X_train and X_test into X_train_normed and X_test_normed
X_train_normed,X_test_normed = X_train.copy(), X_test.copy()

# Specify which column to normalize
col_to_normalize = "Monetary (c.c. blood)"

# Log normalization
for df_ in [X_train_normed, X_test_normed]:
    # Add log normalized column
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    # Drop the original column
    df_.drop(columns=col_to_normalize, inplace=True)

# Check the variance for X_train_normed
X_test_normed.var().round(3)


# In[27]:


from sklearn import linear_model

# Instantiate LogisticRegression
logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)

# Train the model
logreg.fit(X_train_normed, y_train)

# AUC score for tpot model
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')


# In[28]:


from operator import itemgetter

sorted(
    [('tpot', tpot_auc_score), ('logreg', logreg_auc_score)],
    key = itemgetter(1),
    reverse = True
)


# In[29]:


import pickle


# In[31]:


with open('model.bin','wb') as f_out:
    pickle.dump(logreg,f_out)
    f_out.close()


# In[41]:


with open('model.bin','rb') as f_in:
file = pickle.load(f_in)
predi


# In[ ]:




