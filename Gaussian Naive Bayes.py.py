#!/usr/bin/env python
# coding: utf-8

# # Loading the wine dataset

# In[2]:


# importing the necessary libraries and packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import numpy as np
from sklearn.datasets import load_wine


# In[3]:


# loading the whole dataset and converting in into a dataframe

wine = load_wine()

df = pd.DataFrame(wine.data, columns = wine.feature_names)


# In[4]:


df['target'] = wine.target


# In[5]:


df['target_names'] = df.target.apply(lambda x : wine.target_names[x])


# In[6]:


# importing the INDEPENDENT variables from the dataset and storing it into the pandas dataframe

x = pd.DataFrame(sklearn.datasets.load_wine(return_X_y=True)[0], columns = sklearn.datasets.load_wine(return_X_y=False)['feature_names'])


# In[7]:


# importing the DEPENDENT variable from the dataset

y = sklearn.datasets.load_wine(return_X_y=True)[1]


# # Splitting the data into train and test

# In[9]:


# splitting data into train and test data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

print ('training size ', x_train.shape,  y_train.shape)
print ('testing size ', x_test.shape,  y_test.shape)


# # Plotting the class-wise distribution of data

# In[11]:


# plotting the training set
 
plt.bar([0,1,2], np.bincount(y_train), color = 'darkblue')
plt.xticks([0,1,2], sklearn.datasets.load_wine(return_X_y=False)['target_names'])

plt.title('TRAIN SET')
plt.xlabel('Class of Wine')
plt.ylabel('Number of Instances')
 
plt.show()


# plotting the testing set
 
plt.bar([0,1,2], np.bincount(y_test), color = 'crimson')
plt.xticks([0,1,2], sklearn.datasets.load_wine(return_X_y=False)['target_names'])

plt.title('TEST SET')
plt.xlabel('Class of Wine')
plt.ylabel('Number of Instances')
 
plt.show()


# # Training the Gaussian Naive Bayes classifier

# In[13]:


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

# training the model using the training data
model.fit(x_train, y_train)

# making a prediction using the newly trained model
y_pred = model.predict(x_test)

# reporting the class priors
print('the class priors are',model.class_prior_)


# In[14]:


train_df = x_train.copy()


# In[15]:


y_train_target = y_train.copy()
   
train_df['target'] = y_train_target


# In[16]:


print(train_df.groupby(train_df['target']).mean())


# In[17]:


print(train_df.groupby(train_df['target']).var())


# In[21]:


test_df = x_test.copy()


# In[22]:


y_test_target = y_test.copy()
   
test_df['target'] = y_test_target


# In[23]:


print(test_df.groupby(test_df['target']).mean())


# In[24]:


print(test_df.groupby(test_df['target']).var())


# In[25]:


from sklearn import metrics

# checking the accuracy of the model
print("Accuracy is",metrics.accuracy_score(y_test, y_pred))


# In[26]:


# confusion matrix for the model

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)


# In[27]:


plt.figure(figsize=(7,4))

sns.heatmap(cm, annot=True, cmap="BuPu", linewidths=1, linecolor='gold')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()


# # Training the model by setting priors in the ratios:       40-40-20, 33-33-34 and 80-10-10 respectively
# 

# In[28]:


# setting the priors in the ratios 40-40-20

model1 = GaussianNB(priors=[.4,.4,.2])

# training the model using the training data
model1.fit(x_train, y_train)

# making a prediction using the newly trained model
y_pred1 = model1.predict(x_test)


# In[29]:


# checking the accuracy of the model
print("Accuracy is",metrics.accuracy_score(y_test, y_pred1))


# In[30]:


# confusion matrix for the model

cm1 = confusion_matrix(y_test, y_pred1)

print(cm1)


# In[31]:


plt.figure(figsize=(7,4))

sns.heatmap(cm1, annot=True, cmap="YlGnBu", linewidths=1, linecolor='pink')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()


# In[32]:


# setting the priors in the ratios 33-33-34

model2 = GaussianNB(priors=[.33,.33,.34])

# training the model using the training data
model2.fit(x_train, y_train)

# making a prediction using the newly trained model
y_pred2 = model2.predict(x_test)


# In[33]:


# checking the accuracy of the model
print("Accuracy is",metrics.accuracy_score(y_test, y_pred2))


# In[34]:


# confusion matrix for the model

cm2 = confusion_matrix(y_test, y_pred2)

print(cm2)


# In[35]:


plt.figure(figsize=(7,4))

sns.heatmap(cm2, annot=True, cmap="BuPu", linewidths=1, linecolor='gold')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()


# In[36]:


# setting the priors in the ratios 80-10-10

model3 = GaussianNB(priors=[.8,.1,.1])

# training the model using the training data
model3.fit(x_train, y_train)

# making a prediction using the newly trained model
y_pred3 = model3.predict(x_test)


# In[37]:



# checking the accuracy of the model
print("Accuracy is",metrics.accuracy_score(y_test, y_pred3))


# In[38]:


# confusion matrix for the model

from sklearn.metrics import confusion_matrix

cm3 = confusion_matrix(y_test, y_pred3)

print(cm3)


# In[39]:


plt.figure(figsize=(7,4))

sns.heatmap(cm3, annot=True, cmap="YlGnBu", linewidths=1, linecolor='pink')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()


# In[ ]:




