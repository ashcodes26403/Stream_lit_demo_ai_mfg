import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Import to save and load the model

# Set the seed for reproducibility
seed = 42

# Load the Iris dataset (ensure 'iris_dataset.csv' is in the same directory)
iris_df = pd.read_csv("iris_dataset.csv")

# Shuffle the dataset
iris_df = iris_df.sample(frac=1, random_state=seed).reset_index(drop=True)

# Selecting features and target data
X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = iris_df['Species']  # Use Series instead of DataFrame for the target

# Split data into train and test sets (70% training and 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

# Create an instance of the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # Display accuracy to two decimal places

# Save the model to disk
joblib.dump(clf, "rf_model.sav")
print("Model saved to 'rf_model.sav'")

# Load the model from disk (if needed later)
loaded_clf = joblib.load("rf_model.sav")

# Verify the loaded model by making a prediction on the test set
y_pred_loaded = loaded_clf.predict(X_test)

# Confirm that the predictions are the same as the initial model's predictions
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
print(f"Loaded model accuracy: {loaded_accuracy:.2f}")
