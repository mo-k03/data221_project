import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("../../data/kc_house_data.csv")

df = df.drop(columns=["id","date"], errors='ignore')

#fill numerical w mean
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].mean())

#fill categorical w mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

x = df.drop("price", axis=1)
y = df["price"]

#train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.2, random_state= 221
)

#model (gradient boosting)
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=221
)

#train model
model.fit(x_train, y_train)

#predictions
y_pred = model.predict(x_test)

#evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Gradient Boosting Results:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

#feature importance
importances = model.feature_importances_
features = x.columns

indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances (Gradient Boosting)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()
plt.show()
