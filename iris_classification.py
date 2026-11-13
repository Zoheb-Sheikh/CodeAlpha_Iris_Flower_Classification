# ============================================================
# CODEALPHA INTERNSHIP - DATA SCIENCE TASK 1
# PROJECT: IRIS FLOWER CLASSIFICATION
# ============================================================

# ============================================================
# 1. Import Required Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 2. Load Dataset
# ============================================================

df = pd.read_csv("Iris.csv")

print("First Five Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

# ============================================================
# 3. Data Cleaning
# ============================================================

# Remove unnecessary Id column
if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows containing missing values
df.dropna(inplace=True)

# Display dataset statistics
print("\nStatistical Summary:")
print(df.describe())

# ============================================================
# 4. Exploratory Data Analysis (EDA)
# ============================================================

# Display species count
print("\nSpecies Distribution:")
print(df["Species"].value_counts())

# Species count visualization
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Species")
plt.title("Distribution of Iris Flower Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df, hue="Species")
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================================
# 5. Separate Features and Target
# ============================================================

X = df.drop("Species", axis=1)

y = df["Species"]

print("\nInput Features:")
print(X.head())

print("\nTarget Labels:")
print(y.head())

# ============================================================
# 6. Split Dataset into Training and Testing Sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)

# ============================================================
# 7. Build Machine Learning Model
# ============================================================

# StandardScaler scales the input features.
# SVC (Support Vector Classifier) performs classification.

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="linear"))
])

# ============================================================
# 8. Train the Model
# ============================================================

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# ============================================================
# 9. Make Predictions
# ============================================================

y_pred = model.predict(X_test)

print("\nPredicted Species:")
print(y_pred)

print("\nActual Species:")
print(y_test.values)

# ============================================================
# 10. Evaluate Model Performance
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# 11. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("Iris Flower Classification - Confusion Matrix")
plt.tight_layout()
plt.show()

# ============================================================
# 12. Test with New Flower Measurements
# ============================================================

# Example measurements:
# Sepal Length = 5.1
# Sepal Width  = 3.5
# Petal Length = 1.4
# Petal Width  = 0.2

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(new_flower)

print("\nNew Flower Prediction:")
print("Predicted Species:", prediction[0])

# ============================================================
# 13. Save Trained Model
# ============================================================

joblib.dump(model, "iris_classification_model.pkl")

print("\nTrained Model Saved Successfully!")

# ============================================================
# END OF PROJECT
# ============================================================