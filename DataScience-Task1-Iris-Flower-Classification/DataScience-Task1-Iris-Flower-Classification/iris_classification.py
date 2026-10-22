
# Iris Flower Classification
# Oasis Infobyte Data Science Internship

import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,  
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load the Iris dataset wfgtreydhtr
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Dataset Preview:")
print(X.head())

print("\nDataset Shape:", X.shape)
print("\nFlower Classes:", iris.target_names)

# 2. Explore the dataset
print("\nMissing Values:")
print(X.isnull().sum())

# 3. Visualize the data
plt.figure(figsize=(8, 5))
plt.scatter(
    X["petal length (cm)"],
    X["petal width (cm)"],
    c=y
)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Flower Dataset")
plt.savefig("iris_scatter_plot.png")
plt.show()

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Train two machine learning models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=iris.target_names
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(
        name.lower().replace(" ", "_") + "_confusion_matrix.png"
    )
    plt.show()

# 6. Compare models
print("\nModel Comparison:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.2%}")

best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model}")

