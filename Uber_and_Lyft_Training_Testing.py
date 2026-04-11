import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

rides_all = pd.read_csv('cleaned_cab_rides.csv')

rides_all['cab_type'] = LabelEncoder().fit_transform(rides_all['cab_type'])
rides_all['destination'] = LabelEncoder().fit_transform(rides_all['destination'])
rides_all['source'] = LabelEncoder().fit_transform(rides_all['source'])
rides_all['name'] = LabelEncoder().fit_transform(rides_all['name'])

rides = rides_all[["distance", "cab_type", "year", "month", "dayofweek", "hour", "destination", "source", "price", "surge_multiplier", "name"]].dropna(subset=['price'])

X = rides[["distance", "cab_type", "year", "month", "dayofweek", "hour", "destination", "source", "surge_multiplier", "name"]]
y = rides[["price"]]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


linearModel = LinearRegression()
linearModel.fit(X_train, np.ravel(y_train))
linear_pred = linearModel.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knnModel = KNeighborsRegressor(n_neighbors=5)
knnModel.fit(X_train_scaled, np.ravel(y_train))
knn_pred = knnModel.predict(X_test_scaled)

linear_mae = metrics.mean_absolute_error(y_test, linear_pred)
linear_mse = metrics.mean_squared_error(y_test, linear_pred)
linear_rmse = metrics.root_mean_squared_error(y_test, linear_pred)
linear_score = metrics.r2_score(y_test, linear_pred)

knn_mae = metrics.mean_absolute_error(y_test, knn_pred)
knn_mse = metrics.mean_squared_error(y_test, knn_pred)
knn_rmse = metrics.root_mean_squared_error(y_test, knn_pred)
knn_score = metrics.r2_score(y_test, knn_pred)

print(f"Linear MAE: ${linear_mae}\nLinear MSE: ${linear_mse}\nLinear RMSE: ${linear_rmse}\n Linear Score: {linear_score}")
print(f"KNN MAE: ${knn_mae}\nKNN MSE: ${knn_mse}\nKNN RMSE: ${knn_rmse}\n KNN Score: {knn_score}")