import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the datasets
datasets = pd.read_csv('Position_Salaries.csv')
X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:, 2].values

# Reshape Y to a 2D array
Y = Y.reshape(-1, 1)

# Feature Scaling
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
Y_scaled = np.ravel(sc_Y.fit_transform(Y.reshape(-1, 1)))

# Fitting the SVR model to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X_scaled, Y_scaled)

# Predicting a new result with the SVR model
new_position_level = 6.5
scaled_prediction = regressor.predict(sc_X.transform([[new_position_level]]))
predicted_salary = sc_Y.inverse_transform(scaled_prediction.reshape(-1, 1))[0]

# Visualising the SVR Regression results
X_grid = np.arange(min(X_scaled), max(X_scaled), 0.01)  # smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X_scaled, Y_scaled, color='red', label='Actual Data')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Support Vector Regression')
plt.scatter(sc_X.transform([[new_position_level]]), scaled_prediction, color='green', marker='X', s=100,
            label='Predicted Point')
plt.title('Support Vector Regression Results')
plt.xlabel('Position level (Scaled)')
plt.ylabel('Salary (Scaled)')
plt.legend()
plt.show()

print(f'Predicted Salary for Position Level {new_position_level}: ${predicted_salary:.2f}')
# reference from another repository of github and most of parts are copied 
