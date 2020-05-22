## Loading the wine dataset

#importing the necessary libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import numpy as np
from sklearn.datasets import load_wine


#loading the whole dataset and converting in into a dataframe
wine = load_wine()
df = pd.DataFrame(wine.data, columns = wine.feature_names)

df['target'] = wine.target

df['target_names'] = df.target.apply(lambda x : wine.target_names[x])


#importing the INDEPENDENT variables from the dataset and storing it into the pandas dataframe
x = pd.DataFrame(sklearn.datasets.load_wine(return_X_y=True)[0], columns = sklearn.datasets.load_wine(return_X_y=False)['feature_names'])

#importing the DEPENDENT variable from the dataset
y = sklearn.datasets.load_wine(return_X_y=True)[1]





## Splitting the data into train and test

#splitting data into train and test data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
print ('training size ', x_train.shape,  y_train.shape)
print ('testing size ', x_test.shape,  y_test.shape)



## Plotting the class-wise distribution of data

# plotting the training set
plt.bar([0,1,2], np.bincount(y_train), color = 'darkblue')
plt.xticks([0,1,2], sklearn.datasets.load_wine(return_X_y=False)['target_names'])
plt.title('TRAIN SET')
plt.xlabel('Class of Wine')
plt.ylabel('Number of Instances')
 
plt.show()


#plotting the testing set
plt.bar([0,1,2], np.bincount(y_test), color = 'crimson')
plt.xticks([0,1,2], sklearn.datasets.load_wine(return_X_y=False)['target_names'])
plt.title('TEST SET')
plt.xlabel('Class of Wine')
plt.ylabel('Number of Instances')
 
plt.show()




## Training the Gaussian Naive Bayes classifier

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

#training the model using the training data
model.fit(x_train, y_train)

#making a prediction using the newly trained model
y_pred = model.predict(x_test)

#reporting the class priors
print('the class priors are',model.class_prior_)

train_df = x_train.copy()
y_train_target = y_train.copy()
train_df['target'] = y_train_target

print(train_df.groupby(train_df['target']).mean())
print(train_df.groupby(train_df['target']).var())

test_df = x_test.copy()
y_test_target = y_test.copy()
test_df['target'] = y_test_target

print(test_df.groupby(test_df['target']).mean())
print(test_df.groupby(test_df['target']).var())


from sklearn import metrics

#checking the accuracy of the model
print("Accuracy is",metrics.accuracy_score(y_test, y_pred))

#confusion matrix for the model
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(7,4))
sns.heatmap(cm, annot=True, cmap="BuPu", linewidths=1, linecolor='gold')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()




## Training the model by setting priors in the ratios: 40-40-20, 33-33-34 and 80-10-10 respectively

#setting the priors in the ratios 40-40-20
model1 = GaussianNB(priors=[.4,.4,.2])

#training the model using the training data
model1.fit(x_train, y_train)

#making a prediction using the newly trained model
y_pred1 = model1.predict(x_test)

#checking the accuracy of the model
print("Accuracy is",metrics.accuracy_score(y_test, y_pred1))

#confusion matrix for the model
cm1 = confusion_matrix(y_test, y_pred1)
print(cm1)

plt.figure(figsize=(7,4))
sns.heatmap(cm1, annot=True, cmap="YlGnBu", linewidths=1, linecolor='pink')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

#setting the priors in the ratios 33-33-34
model2 = GaussianNB(priors=[.33,.33,.34])

#training the model using the training data
model2.fit(x_train, y_train)

#making a prediction using the newly trained model
y_pred2 = model2.predict(x_test)

#checking the accuracy of the model
print("Accuracy is",metrics.accuracy_score(y_test, y_pred2))

#confusion matrix for the model
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

plt.figure(figsize=(7,4))
sns.heatmap(cm2, annot=True, cmap="BuPu", linewidths=1, linecolor='gold')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

#setting the priors in the ratios 80-10-10
model3 = GaussianNB(priors=[.8,.1,.1])

#training the model using the training data
model3.fit(x_train, y_train)

#making a prediction using the newly trained model
y_pred3 = model3.predict(x_test)

#checking the accuracy of the model
print("Accuracy is",metrics.accuracy_score(y_test, y_pred3))

#confusion matrix for the model
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)
print(cm3)

plt.figure(figsize=(7,4))
sns.heatmap(cm3, annot=True, cmap="YlGnBu", linewidths=1, linecolor='pink')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()



#distribution of alcohol content by class
for i in df.target.unique():
    sns.distplot(df['alcohol'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()


plt.figure(1)                       #distribution of malic_acid content by class
for i in df.target.unique():
    sns.distplot(df['malic_acid'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(2)                    #distribution of ash content by class
for i in df.target.unique():
    sns.distplot(df['ash'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(3)                     #distribution of alcalinity_of_ash content by class
for i in df.target.unique():
    sns.distplot(df['alcalinity_of_ash'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(4)                  #distribution of magnesium content by class
for i in df.target.unique():
    sns.distplot(df['magnesium'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(5)                  #distribution of total_phenols content by class
for i in df.target.unique():
    sns.distplot(df['total_phenols'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(6)                 #distribution of flavanoids content by class
for i in df.target.unique():
    sns.distplot(df['flavanoids'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(7)                #distribution of nonflavanoids_phenols content by class
for i in df.target.unique():
    sns.distplot(df['nonflavanoid_phenols'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(8)               #distribution of proanthocyanins content by class
for i in df.target.unique():
    sns.distplot(df['proanthocyanins'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(9)                #distribution of color_intensity content by class
for i in df.target.unique():
    sns.distplot(df['color_intensity'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(10)              #distribution of hue content by class
for i in df.target.unique():
    sns.distplot(df['hue'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(11)             #distribution of od280/od315_of_diluted_wines content by class
for i in df.target.unique():
    sns.distplot(df['od280/od315_of_diluted_wines'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()

plt.figure(12)            #distribution of proline content by class
for i in df.target.unique():
    sns.distplot(df['proline'][df['target']==i], kde=1,label='{}'.format(i))
plt.legend()


