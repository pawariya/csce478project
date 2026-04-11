import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# LOADING AND PREPROCESSING DATA

rides_all = pd.read_csv('cleaned_cab_rides.csv')

rides_all['cab_type'] = LabelEncoder().fit_transform(rides_all['cab_type'])
rides_all['destination'] = LabelEncoder().fit_transform(rides_all['destination'])
rides_all['source'] = LabelEncoder().fit_transform(rides_all['source'])
rides_all['name'] = LabelEncoder().fit_transform(rides_all['name'])

rides = rides_all[["distance", "cab_type", "year", "month", "dayofweek", "hour", "destination", "source", "price", "surge_multiplier", "name"]].dropna(subset=['price'])

X = rides[["distance", "cab_type", "year", "month", "dayofweek", "hour", "destination", "source", "surge_multiplier", "name"]]
y = rides[["price"]]

#TRAIN & TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#MAKING PREDICTIONS

linearModel = LinearRegression()
linearModel.fit(X_train, np.ravel(y_train))
linear_pred = linearModel.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knnModel = KNeighborsRegressor(n_neighbors=5)
knnModel.fit(X_train_scaled, np.ravel(y_train))
knn_pred = knnModel.predict(X_test_scaled)

# STANDARD REGRESSION METRICS

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

#flattening y_test for price from dataframe to series
y_true = y_test.values.flatten()

# ERROR STATS - base for comparison of models

def error_stats(y_true, y_pred):
    errors = np.abs(y_true - y_pred) # if positive, model underpredicted, else it overpredicted
    mean_error = np.mean(errors)
    variance_error = np.var(errors)
    return mean_error, variance_error

linear_mean_error, linear_variance_error = error_stats(y_true, linear_pred)
knn_mean_error, knn_variance_error = error_stats(y_true, knn_pred)

print("Error Statistics")
print(f"Linear Mean Absolute Error: {linear_mean_error}")
print(f"Linear Error Variance: {linear_variance_error}\n")

print(f"KNN Mean Absolute Error: {knn_mean_error}")
print(f"KNN Error Variance: {knn_variance_error}")

# VISUAL COMPARISONS

models = ['Linear Regression', 'KNN']

mae_values = [linear_mae, knn_mae]
rmse_values = [linear_rmse, knn_rmse]
variance_values = [linear_variance_error, knn_variance_error]
r2_values = [linear_score, knn_score]

x = np.arange(len(models))
width = 0.25

# MAE, RMSE, Variance
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width, mae_values, width, label='MAE')
bars2 = ax.bar(x, rmse_values, width, label='RMSE')
bars3 = ax.bar(x + width, variance_values, width, label='Variance')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Value')
ax.set_title('Model Comparison: Error Metrics')
ax.legend()

#labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()

# R^2
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(models, r2_values)
ax.set_ylabel('R^2 Score')
ax.set_title('Model Comparison: R^2')

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
