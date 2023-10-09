import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


import matplotlib.style

titan_traindf = pd.read_csv('train.csv')

titan_testdf=pd.read_csv('test.csv')

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
titan_traindf.info()

titan_traindf.describe()

titan_traindf.describe()

titan_traindf.head()

titan_testdf.head()

titan_traindf.isna()

titan_testdf.isna()

##Lets look how may people survived divided by class
sns.countplot(x='Survived', hue='Pclass', data= titan_traindf)

sns.boxplot(x='Pclass', y='Age', data=titan_traindf)

# We can see that there are few value for which value null which is not possible there we replace such values with average age
titan_traindf['Age']=titan_traindf['Age'].fillna(titan_traindf['Age'].mean())
titan_traindf

titan_traindf.isna()

# Replacing null value present in Age column with the average age
titan_testdf['Age']=titan_testdf['Age'].fillna(titan_testdf['Age'].mean())
titan_testdf

titan_testdf.isna()

titan_traindf.drop("Cabin",inplace=True,axis=1)
pd.get_dummies(titan_traindf["Sex"])
sex = pd.get_dummies(titan_traindf["Sex"],drop_first=True)

embarked = pd.get_dummies(titan_traindf["Embarked"],drop_first=True)
pclass= pd.get_dummies(titan_traindf["Pclass"],drop_first=True)

titan_train = pd.concat([titan_traindf,pclass,sex,embarked],axis=1)

titan_train

titan_train.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked"],axis=1,inplace=True)

titan_train

X = titan_train.drop("Survived",axis=1)
y = titan_train["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
