# Linear Regression Model - Mohamad Khierandish

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from utils.common import dataCleaning, trainTestSplit80_20

# preprocesses data
dataCleaning("../../data/kc_house_data.csv")

# data loading
kch_features_train, kch_features_test, kch_target_train, kch_target_test = trainTestSplit80_20("../../data/kc_house_data_cleaned.csv")

# train the model
linear_regression_model = LinearRegression()
linear_regression_model.fit(kch_features_train, kch_target_train)

# make predictions
predicted_prices = linear_regression_model.predict(kch_features_test)

# performance analysis
mae_value = mean_absolute_error(kch_target_test, predicted_prices)
rmse_value = np.sqrt(mean_squared_error(kch_target_test, predicted_prices))
r2_value = r2_score(kch_target_test, predicted_prices)

print("Linear Regression Results: ")
print("MAE:", mae_value)
print("RMSE:", rmse_value)
print("R^2:", r2_value)

# create feature-coefficient table
df = pd.read_csv("../../data/kc_house_data_cleaned.csv")
kch_features = df.drop("price", axis=1)
feature_coefficients = pd.DataFrame({
    "Features": kch_features.columns,
    "Coefficient": linear_regression_model.coef_
})

# create absolute values
feature_coefficients["Absolute_Coefficients"] = feature_coefficients["Coefficient"].abs()

# normalizing the coefficients
feature_coefficients["Normalized_Importance"] = (
        feature_coefficients["Absolute_Coefficients"] / feature_coefficients["Absolute_Coefficients"].sum()
)

# rank from most important to least
feature_coefficients = feature_coefficients.sort_values(by="Normalized_Importance", ascending=False)

# plot it as a bar graph
plt.figure()
plt.bar(feature_coefficients["Features"], feature_coefficients["Normalized_Importance"])    # x-axis is feature names, y-axis is the abs. coef. of it

plt.xticks(rotation=90)     # rotate the x-axis name so they don't overlap each other
plt.xlabel("Features")      # x-axis label
plt.ylabel("Importance")    # y-axis label

plt.title("Feature Importance for Linear Regression")   # title

plt.tight_layout()  # adjust the spacing so the data fits nicely in the graph
plt.show()
