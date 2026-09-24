from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

wine = load_wine()
feature = wine.data
label = wine.target

df = pd.DataFrame(feature, columns=wine.feature_names)
df['Wine Type'] = [wine.target_names[i] for i in label]

print("Data sum: ", feature.shape[0])
print("Feature sum: ", feature.shape[1])
print("Class type: ", wine.target_names)
print()
df.head(10)

feature_train, feature_test, label_train, label_test = train_test_split(
    feature, label, test_size=0.2, random_state=42
)
scaler = StandardScaler()
scaler.fit(feature_train)
feature_train = scaler.transform(feature_train)
feature_test = scaler.transform(feature_test)
print("Sum of training data:", feature_train.shape[0])
print("Sum of test data:", feature_test.shape[0])

model = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
model.fit(feature_train, label_train)
print("Iteration amount:", model.n_iter_)

label_pred = model.predict(feature_test)
accuracy = accuracy_score(label_test, label_pred)
print(f"Model accuracy: {accuracy * 100:.1f}%")
print("Detail for each class:")
print(classification_report(label_test, label_pred, target_names=wine.target_names))

hasil = pd.DataFrame(feature_test, columns=wine.feature_names)
hasil['Wine Type'] = [wine.target_names[i] for i in label_test]
hasil['Predicted Wine Type'] = [wine.target_names[i] for i in label_pred]
hasil['True?'] = hasil['Wine Type'] == hasil['Predicted Wine Type']

hasil.head(10)
