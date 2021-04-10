# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

titanic_data = pd.read_csv('titanic.txt')
print(titanic_data.head(10))
# The number of passengers
print('Number of passengers:'+str(len(titanic_data.index)))

# Analyzing the dataset

# Plot of survivors
survivors = sns.countplot(x='Survived', data=titanic_data)
print(survivors)
# Plot of gender survivors
gender_survivors = sns.countplot(x='Survived', hue='Sex', data=titanic_data)
plt.show()
# Plot of male to female
gender = sns.countplot(x='Sex', data=titanic_data)
plt.show()
# Plot of survivors based on class
survivors_class = sns.countplot(x='Survived', hue='Pclass', data=titanic_data)
plt.show()
# Age distribution
Age = titanic_data['Age'].plot.hist()
plt.show()
# Fare Analysis
Fare = titanic_data['Fare'].plot.hist(bins=20, figsize=(10, 5))
plt.show()
print(titanic_data.info())
Sib_sp = sns.countplot(x='Survived', hue='SibSp', data=titanic_data)
plt.show()

# Data Wrangling

print(titanic_data.isnull().sum())
# Checking the missing values column using heatmap
missing_values = sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap='viridis')
plt.show()
# Analyze the relationship of age to the class
age_class = sns.boxplot(x='Pclass', y='Age', data=titanic_data)
plt.show()
# Remove the cabin column due to the high number of missing values
cabin_column = titanic_data.drop('Cabin', axis=1, inplace=True)
print(cabin_column)
print(titanic_data.head(5))
# Remove the Nan values in the dataset
Nan_values = titanic_data.dropna(inplace=True)
print(Nan_values)
# Checking the missing values column using heatmap
missing_values2 = sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap='viridis')
plt.show()
# Replacing the sex column to a categorical data
sex = pd.get_dummies(titanic_data['Sex'], drop_first=True)
print(sex)
# Replacing the embarked column to a categorical data
embark = pd.get_dummies(titanic_data['Embarked'], drop_first=True)
print(embark)
# Replacing the Pclass column to a categorical data
Pcl = pd.get_dummies(titanic_data['Pclass'], drop_first=True)
print(Pcl)
# Concatenate the data together
titanic_data = pd.concat([titanic_data, sex, embark, Pcl], axis=1)
print(titanic_data)
# Drop the unnecessary column
drop_column = titanic_data.drop(['Name', 'Sex', 'Pclass', 'Ticket', 'Embarked', 'PassengerId'], axis=1, inplace=True)
print(drop_column)
print(titanic_data.head(5))

# Building a model: Test and training set

X = titanic_data.drop('Survived', axis=1)
Y = titanic_data['Survived']
# Splitting the dataset into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
# Fitting Logistic Regression into Training set.
logmodel = LogisticRegression(random_state=1, solver='lbfgs')
logmodel.fit(X_train, Y_train)
# Predict the results
predictions = logmodel.predict(X_test)

# Accuracy check
# Use classification report
print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(accuracy_score(Y_test, predictions))


