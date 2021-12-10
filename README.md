# 7 Random Forest vs. Support Vector Machine Algorithm -- Classification

In this project, the titanic dataset on kaggle, https://www.kaggle.com/c/titanic/data, was used to build a random forest and support vector machine classifiers. The dataset includes each passenger's details such as his/her age, sex, the cabin class, name, fare, ticket, etc. Each passenger destiny is labelled as either 0 or 1 to indicate whether or not the passenger survived the disaster. In this project, the three features of age, sex, and the cabin class were chosen as the independent variables. The passenger destiny is predicted through either random forest or support vector machine. 

## Understanding the data:
The chosen features, passengers' age, sex, and cabin class, were visualized in the following figure.

![fig](https://user-images.githubusercontent.com/54812742/145497452-fae983a4-9212-4d79-ac58-8b21ba7aa008.PNG)

## RF vs. SVM algorithm
The labelled dataset was splitted: 30 % was put aside for testing and the remaining 70 % was used to train the model. The metrics was used to evaluate the model performance. The RF algorithm with 83.2 % accuracy outperforms the SVM model with 61.4 % accuracy. To understand the relative importance of each feature, the feature importance for the RF algorithm was calculated. As shown in the following table, the passenger's age contributes the most on the passengers' destiny. 

![table](https://user-images.githubusercontent.com/54812742/145500371-707e6873-36e9-4317-b3f3-a2af0ced3f5a.PNG)
