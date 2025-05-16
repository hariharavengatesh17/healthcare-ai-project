import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the expanded dataset
df = pd.read_csv("expanded_healthcare_dataset.csv")

# Features and label
X = df.drop("disease", axis=1)
y = df["disease"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "diagnosis_model_v2.pkl")
print("✅ Model trained and saved as diagnosis_model_v2.pkl")
