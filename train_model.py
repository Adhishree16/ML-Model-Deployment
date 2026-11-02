from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Create the machine learning model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Step 3: Train the model
model.fit(X, y)

# Step 4: Save the trained model
joblib.dump(
    {
        "model": model,
        "target_names": iris.target_names.tolist()
    },
    "model.pkl"
)

print("Model trained successfully!")
print("Model saved as model.pkl")