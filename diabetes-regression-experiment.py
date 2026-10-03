import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import pandas as pd

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

df = pd.DataFrame(X, columns=diabetes.feature_names)
df["Progression"] = y

print("Data sum: ", X.shape[0])
print("Feature sum: ", X.shape[1])
print("Class type: Regression")
print()
df.head(10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Trained data sum: ", X_train.shape[0])
print("Test data sum: ", X_test.shape[0])

experiment_model = MLPRegressor(
    hidden_layer_sizes=(10, 5, 4),
    activation="relu",
    max_iter=1000,
    random_state=42,
    learning_rate_init=0.001,
    alpha=0.0010
)

experiment_model.fit(X_train, y_train)

print("Experiment model has finished training.")
print("Number of iteration used: ", experiment_model.n_iter_)

y_pred_experiment = experiment_model.predict(X_test)

experiment_r2_percent = r2_score(y_test, y_pred_experiment)
print(f"R2 Score Experiment Model: {experiment_r2_percent*100}%")

y_pred_exp = experiment_model.predict(X_test)

result_exp = pd.DataFrame(X_test, columns=diabetes.feature_names)
result_exp["Actual_Answer"] = y_test
result_exp["Model_Guess"] = y_pred_exp
result_exp["Error_Diff"] = np.abs(
    result_exp["Actual_Answer"] - result_exp["Model_Guess"])

result_exp.head(15)
