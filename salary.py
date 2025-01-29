import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('Salary_Data.csv')

X = df.iloc[:, :-1].values
X = X.reshape(-1, 1)  # Reshape X to be a 2D array
Y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.linear_model import LinearRegression
r = LinearRegression()
r.fit(X_train, Y_train)

# Save the LinearRegression model
pickle.dump(r, open('salary.pkl', 'wb'))