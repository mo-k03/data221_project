from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd
from data_preprocessing_basic import clean_data

# preprocesses data
clean_data('kc_house_data.csv')

# data loading
data = pd.read_csv('kc_house_data_cleaned.csv')
feature_names = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','lat','long','sqft_living15','sqft_lot15','age_at_sale']
feature_matrix = data.loc[:,feature_names]
target_labels = data.loc[:,'price']

# training split, 80/20
features_train, features_test, labels_train, labels_test = (
    train_test_split(feature_matrix, target_labels, test_size=0.2, random_state=42)) # 80/20 split

# data standardization
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# model creation
decision_tree_regressor = DecisionTreeRegressor(criterion='squared_error', min_impurity_decrease = 0.01, min_samples_split= 15, max_depth=30, random_state=42)
decision_tree_regressor.fit(features_train, labels_train)

# model training accuracy
predicted_train = decision_tree_regressor.predict(features_train)
r2_train = r2_score(labels_train, predicted_train)

# model test accuracy
predicted_test = decision_tree_regressor.predict(features_test)
r2_test = r2_score(labels_test, predicted_test)

# calculate Mean Absolute Error (The average $ amount you are off by)
mae_test = mean_absolute_error(labels_test, predicted_test)

print(f"Training R2 Score: {r2_train:.4f}")
print(f"Testing R2 Score: {r2_test:.4f}")
print(f"Average Error: ${mae_test:,.2f}")

importances = decision_tree_regressor.feature_importances_
feature_names = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','lat','long','sqft_living15','sqft_lot15','age_at_sale'
]
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

