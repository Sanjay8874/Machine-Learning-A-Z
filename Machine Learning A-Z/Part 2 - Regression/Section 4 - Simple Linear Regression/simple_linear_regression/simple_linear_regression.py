#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#Splitting the data into training set and test set
dataset = pd.read_csv('salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values



#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)


#Fitting the simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, y_train)



#Presict the set result
y_pred = regressor.predict(X_test)

#visualising Traing set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue' )
plt.title('Experiance vs salary')
plt.xlabel('year of Experiance')
plt.ylabel('Salary')
plt.show()


#visualising tesing set result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue' )
plt.title('Experiance vs salary')
plt.xlabel('year of Experiance')
plt.ylabel('Salary')
plt.show()
