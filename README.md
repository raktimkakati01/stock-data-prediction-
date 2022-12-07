# stock-data-prediction-

This code first loads the stock data from a CSV file using the pandas library. It then extracts the features (i.e. the data used to make predictions) and the target (i.e. the data we want to predict) from the data.

Next, the code uses the train_test_split function from the sklearn library to split the data into training and testing sets. The training set is used to train the machine learning model, while the testing set is used to evaluate the performance of the model.

The code then creates a linear regression model using the LinearRegression class from sklearn, and trains the model on the training data using the fit method. Once the model is trained, it can be used to make predictions on new data using the predict method.

Finally, the code uses the trained model to make predictions on the testing data, and calculates the mean absolute error (MAE) of the predictions using the numpy library. It then uses the trained model to make a prediction on a hypothetical stock with certain features.
