import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
data = pd.read_csv("BlackFridaySales.csv")

# Data Preprocessing
data['Product_Category_2'] = data['Product_Category_2'].fillna(0).astype('int64')
data['Product_Category_3'] = data['Product_Category_3'].fillna(0).astype('int64')

# Encoding categorical variables
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Age'] = le.fit_transform(data['Age'])
data['City_Category'] = le.fit_transform(data['City_Category'])

# Dummy variables for Stay_In_Current_City_Years
data = pd.get_dummies(data, columns=['Stay_In_Current_City_Years'])

# Drop irrelevant columns
data = data.drop(["User_ID", "Product_ID"], axis=1)

# Splitting data into independent and dependent variables
X = data.drop("Purchase", axis=1)
y = data['Purchase']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))

# Random Forest Regressor
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# XGBoost Regressor
xgb = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))