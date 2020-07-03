import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

from linear_regression import LinearRegression

# load data
FILE_NAME = "data.csv"
data = np.genfromtxt(FILE_NAME, delimiter=",", dtype=np.float32, skip_header=1)

# split data
n_samples, n_features = data.shape
n_features -= 1
X = data[:, 0:n_features]
y = data[:, n_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit
model = LinearRegression(X_train, y_train)
model.fit()

# plot
y_pred_line = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
scat = plt.scatter(X, y, color=cmap(0.9), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()

# accuracy
train_accuracy = model.score()
test_accuracy = model.score(X_test, y_test)

table_accuracy = pd.DataFrame([
				[train_accuracy],
				[test_accuracy]],
				['Training Accuracy', 'Test Accuracy'],	
				['Linear Regression'])
print(table_accuracy)

# parameters
params = model.get_params()
theta_0 = params[0, 0]
theta_1 = params[1, 0]
table_params = pd.DataFrame([
				[theta_0],
				[theta_1]],
				['theta_0', 'theta_1'],	
				['Parameters'])
print(table_params)

# save the model
filename = 'model'
pickle.dump(model, open(filename, 'wb'))