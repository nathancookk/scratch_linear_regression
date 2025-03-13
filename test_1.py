import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=1, random_state=23)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

fig = plt.figure(figsize=(10, 8))
plt.scatter(X, y, color='green', marker='o', s=10)
plt.show()

regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

mse = mse(y_test, predictions)
print("MSE:", mse)

y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(10, 8))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()