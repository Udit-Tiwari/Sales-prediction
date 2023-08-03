# Sales-prediction
#It is a sales prediction AI project using python programming language
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Sample sales dataset (use your actual dataset for better results)
data = {
    'Advertising': [100, 200, 300, 400, 500],
    'Price': [5, 6, 7, 8, 9],
    'Promotion': [20, 30, 25, 35, 40],
    'Sales': [150, 300, 450, 550, 700]
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Extract features (X) and target variable (y)
X = df[['Advertising', 'Price', 'Promotion']]
y = df['Sales']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the random forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict sales for the test set
predicted_sales = model.predict(X_test)

# Evaluate the model's performance (optional)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predicted_sales)
r2 = r2_score(y_test, predicted_sales)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Predict sales for new advertising values
new_data = {
    'Advertising': [600, 756, 856],
    'Price': [9, 8, 7],
    'Promotion': [40, 35, 30]
}
new_df = pd.DataFrame(new_data)
predicted_new_sales = model.predict(new_df)

# Output the predicted sales for the new advertising values
for ad, sales in zip(new_data['Advertising'], predicted_new_sales):
    print(f"Predicted sales for advertising ${ad}: ${sales:.2f}")

