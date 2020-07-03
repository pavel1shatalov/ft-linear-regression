import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from linear_regression import LinearRegression

# load the model
try:
	file_model = 'model'
	model = pickle.load(open(file_model, 'rb'))
except:
	# load data
	file_data = "data.csv"
	data = np.genfromtxt(file_data, delimiter=",", dtype=np.float32, skip_header=1)

	# split data
	n_samples, n_features = data.shape
	n_features -= 1
	X = data[:, 0:n_features]
	y = data[:, n_features]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# init model
	model = LinearRegression(X_train, y_train)

# prompt
x = float(input())

# prepare data
X = np.reshape(x, (-1, 1))

# predict
y = model.predict(X)
result = round(y[0,0], 3)

print(result)