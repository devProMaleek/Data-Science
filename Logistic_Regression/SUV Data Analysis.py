import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Importing the dataset
suv_dataset = pd.read_csv('suv_data.csv')
print(suv_dataset.head(10))
print(suv_dataset.isnull().sum())
sns.heatmap(suv_dataset.isnull(), yticklabels=False, cmap='viridis')
plt.show()
# Defining X and Y
X = suv_dataset.iloc[:, [2, 3]].values
Y = suv_dataset.iloc[:, 4].values
print(X)
print(Y)
# Splitting the dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
# Feature Scaling: Scaling the dataset down
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Logistic Regression in to Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)
# Predict the model
Y_pred = classifier.predict(X_test)
print(X_test)
# Checking the accuracy score
print(classification_report(Y_pred, Y_test))
print(accuracy_score(Y_pred, Y_test))
print(confusion_matrix(Y_pred, Y_test))
