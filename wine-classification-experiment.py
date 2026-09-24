from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

wine = load_wine()
feature = wine.data
label = wine.target

feature_train, feature_test, label_train, label_test = train_test_split(
    feature, label, test_size=0.2, random_state=42
)
scaler = StandardScaler()
scaler.fit(feature_train)
feature_train = scaler.transform(feature_train)
feature_test = scaler.transform(feature_test)

experiments = [
    {
        "hidden_layer_sizes": (32,),
        "activation": "relu",
        "max_iter": 500
    },
    {
        "hidden_layer_sizes": (64,),
        "activation": "relu",
        "max_iter": 500
    },
    {
        "hidden_layer_sizes": (128,),
        "activation": "relu",
        "max_iter": 500
    },
    {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "max_iter": 500
    },
    {
        "hidden_layer_sizes": (64,),
        "activation": "tanh",
        "max_iter": 500
    },
    {
        "hidden_layer_sizes": (64,),
        "activation": "logistic",
        "max_iter": 500
    }
]

experiment_result = []

for i, params in enumerate(experiments, start=1):

    model = MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        solver="adam",
        max_iter=params["max_iter"],
        random_state=42
    )

    model.fit(feature_train, label_train)

    label_pred = model.predict(feature_test)

    accuracy = accuracy_score(label_test, label_pred)

    experiment_result.append({
        "Experiment": i,
        "Hidden Layer": str(params["hidden_layer_sizes"]),
        "Activation": params["activation"],
        "Max Iteration": params["max_iter"],
        "Actual Iteration": model.n_iter_,
        "Accuracy": accuracy * 100
    })

df_result = pd.DataFrame(experiment_result)

print("Experiment Result")
print(df_result.to_string(index=False))
