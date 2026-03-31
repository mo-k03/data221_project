import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
from utils.common import dataCleaning, trainTestSplit80_20

# preprocesses data
dataCleaning('../../data/kc_house_data.csv')
features_train, features_test, labels_train, labels_test = trainTestSplit80_20('../../data/kc_house_data_cleaned.csv')

# model creation
decision_tree_regressor = DecisionTreeRegressor(criterion='squared_error',
                                                min_impurity_decrease = 0.05,
                                                min_samples_split= 25,
                                                max_depth=40,
                                                min_samples_leaf=10,
                                                random_state=221)
decision_tree_regressor.fit(features_train, labels_train)

# model training accuracy
predicted_train = decision_tree_regressor.predict(features_train)
r2_train = r2_score(labels_train, predicted_train)

# model test accuracy
predicted_test = decision_tree_regressor.predict(features_test)
r2_test = r2_score(labels_test, predicted_test)

# calculate Mean Absolute Error (The average $ amount you are off by)
mae_test = mean_absolute_error(labels_test, predicted_test)
# calculate Squared Mean Error
mse = mean_squared_error(labels_test, predicted_test)

# model accuracy metrics
print(f"Training R2 Score: {r2_train:.4f}")
print(f"Testing R2 Score: {r2_test:.4f}")
print(f"Mean Absolute Error: ${mae_test:,.2f}")
print(f"Mean Squared Error: ${mse:.4f}")
print(f"Sqrt of MSE: ${np.sqrt(mse):.4f}")

#feature importance
importances = decision_tree_regressor.feature_importances_
df = pd.read_csv("../../data/kc_house_data_cleaned.csv")
x = df.drop("price", axis=1)
features = x.columns

indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances (Decision Tree)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()
plt.show()
