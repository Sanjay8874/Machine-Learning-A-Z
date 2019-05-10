#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#Fitting the regretion model to data set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)


#Predict new result with polynomeal regression
y_pred = regressor.predict(6.5)

#visualisation the regression result
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualisation the Regression result (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 00.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
