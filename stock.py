# Import the necessary libraries

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the stock data
data = pd.read_csv('stock_data.csv')

# Extract the features and target
X = data[['Open', 'Close', 'Volume']]  # features
y = data['Profit']  # target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a linear regression model
model = LinearRegression()


# Train the model on the training data
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean absolute error of the predictions
mae = np.mean(np.abs(y_pred - y_test))
print('Mean Absolute Error:', mae)

# Use the trained model to predict the profit of a hypothetical stock
# with the following features: Open=100, Close=110, Volume=1000
X_new = [[100, 110, 1000]]
new_profit_pred = model.predict(X_new)
print('Predicted profit:', new_profit_pred[0])
