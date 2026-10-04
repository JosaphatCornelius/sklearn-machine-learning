import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# Load data
new_df = pd.read_csv('./datasets/Food_Delivery_Times.csv')

# Define X and y
X = new_df.drop(columns=['Order_ID', 'Delivery_Time_min'])
y = new_df['Delivery_Time_min']

# Encode categorical variables to prevent StandardScaler errors
X = pd.get_dummies(X, drop_first=True)

print("Data sum: ", X.shape[0])
print("Feature sum: ", X.shape[1])
print("Class type: Regression")
print()
print(new_df.head(10))

# Handle missing (NaN) values
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Trained data sum: ", X_train.shape[0])
print("Test data sum: ", X_test.shape[0])

# Train MLP Regressor
model_delivery = MLPRegressor(
    hidden_layer_sizes=(20, 10),
    activation="relu",
    max_iter=500,
    random_state=42
)
model_delivery.fit(X_train, y_train)

print("Model has finished training.")
print("Number of iterations used: ", model_delivery.n_iter_)

# Predict and evaluate
y_pred = model_delivery.predict(X_test)
r2_percent = r2_score(y_test, y_pred)
print(f"R2 Score Model: {r2_percent * 100:.2f}%")

# Properly structure results DataFrame
result = pd.DataFrame({
    'Actual_Answer': y_test.values,
    'Model_Guess': y_pred
})
result["Error_Diff"] = np.abs(result["Actual_Answer"] - result["Model_Guess"])
print(result.head(15))
