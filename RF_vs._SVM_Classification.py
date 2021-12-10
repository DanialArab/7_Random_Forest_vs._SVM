"""
Created on Thu Dec  9 11:42:15 2021

@author: Danial Arab
"""
# Step_1: Importing the required libraries 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.svm import LinearSVC

# Step_2: Data importing and handling 

df = pd.read_csv('train.csv')
size = df['Sex'].value_counts()
print(size)

data = df[['Survived', 'Pclass', 'Sex', 'Age']]
data = data.dropna()

data.Sex[data.Sex == 'male'] = 1
data.Sex[data.Sex == 'female'] = 2

Y = data['Survived'].values
X = data.drop(['Survived' ], axis = 1)

# Step_3: Splitting the data 

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.3, random_state = 20)

# STep_4: Random forest vs. SVM algorithm 

model_rf = RandomForestClassifier(n_estimators = 10, random_state = 40)

model_svm = LinearSVC(max_iter = 1000, random_state = 40)

# Step_5: Training the model and then making predictions 

model_rf.fit (X_train, Y_train)
prediction = model_rf.predict (X_test)

model_svm.fit (X_train, Y_train)
prediction = model_svm.predict (X_test)

# Step_6: Model evaluation 

print('Accuracy = ', metrics.accuracy_score(Y_test, prediction))

# Step_7: Identifying the most influential features 

feature_list = list(X.columns)
feature_importance = pd.Series (model_rf.feature_importances_, index = feature_list).sort_values(ascending = False)
print(feature_importance)
