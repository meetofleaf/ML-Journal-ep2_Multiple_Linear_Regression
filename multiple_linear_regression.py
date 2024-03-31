### Import libraries
import numpy as np
import pandas as pd


### Import Dataset
dataset = pd.read_csv('50_Startups.csv')    # Replace <file_path> with the file path of the dataset within the quotes.


# Extracting independent (X) and dependent (y) columns/fields/variables. You can experiment with different independent variables.
X = dataset.iloc[:,:-1].values      # Independent variables (R&D, Admin, Marketing, State)
y = dataset.iloc[:,-1].values       # Dependent variable (profit)
# Reference for iloc: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html


# Encode categorical variable (state) into dummy variables (in the form of 1s & 0s) so that it can be used in mathematical formula of regression.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# There is a precaution that usually needs to be taken to avoid something called dummy variable trap.
# Luckily, our class and methods from sklearn automatically takes care of it. Visit the link to learn more: https://www.learndatasci.com/glossary/dummy-variable-trap/
# To know more about dummy variables or one hot encoding visit: https://www.shiksha.com/online-courses/articles/handling-categorical-variables-with-one-hot-encoding/


### Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


### Training the Model
from sklearn.linear_model import LinearRegression

# Training the Linear Regression model (named regressor here) on the training set
regressor = LinearRegression()      # Define regressor as regression instance. The linear regression class works for multiple regression as well.
regressor.fit(X_train, y_train)     # Train the model on training data of both independent and dependent data.


### Prediction
# Function to predict profit based on input using our trained model (regressor)
y_pred = regressor.predict(X_test)
# Simply, after learning from training data, the model tries to predict the dependent data (y_pred) from independent data (X_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),axis=1))

### If you were able to run till here successfully, then machine learning part is complete.

# Now, we can't visualize the data as we 1 dimension per variable,
# we have 5 variables so it's not possible to visualize and perceive 5 dimensional charts so we just utilize the predict output above to compare actual and predicted values.
