import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------------
# Part 1: Dataset Creation (Imbalanced)
# -------------------------------

# 15 Positive (1), 5 Negative (0)
actual = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0]

# Predicted labels (some errors added intentionally)
predicted = [1,1,1,0,1,1,0,1,1,0,1,1,1,0,1, 0,0,1,0,0]

df = pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted
})

print("Dataset:\n")
print(df)

# -------------------------------
# Part 2: Accuracy
# -------------------------------

# Manual Accuracy
correct = sum([1 for i in range(len(actual)) if actual[i] == predicted[i]])
accuracy_manual = correct / len(actual)

print("\nManual Accuracy:", accuracy_manual)

# Library Accuracy
accuracy_lib = accuracy_score(actual, predicted)
print("Library Accuracy:", accuracy_lib)

# -------------------------------
# Part 3: Confusion Matrix
# -------------------------------

cm = confusion_matrix(actual, predicted)
TN, FP, FN, TP = cm.ravel()

print("\nConfusion Matrix:\n", cm)

print("\nValues:")
print("True Positive (TP):", TP)
print("True Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)

# Tabular format
cm_df = pd.DataFrame(
    cm,
    index=["Actual Negative", "Actual Positive"],
    columns=["Predicted Negative", "Predicted Positive"]
)

print("\nConfusion Matrix Table:\n")
print(cm_df)

# -------------------------------
# Part 4: Precision & Recall
# -------------------------------

precision = precision_score(actual, predicted)
recall = recall_score(actual, predicted)

print("\nPrecision:", precision)
print("Recall:", recall)

# -------------------------------
# Part 5: F1 Score
# -------------------------------

f1 = f1_score(actual, predicted)

print("F1 Score:", f1)

# -------------------------------
# Part 7: Visualization
# -------------------------------

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# Part 6 & Final Analysis
# -------------------------------

print("\n--- Final Analysis ---")

print("\n1. Effect of increasing False Positives:")
print("-> Precision decreases")

print("\n2. Effect of increasing False Negatives:")
print("-> Recall decreases")

print("\n3. Important Metric in Disease Detection:")
print("-> Recall (to avoid missing actual cases)")

print("\n4. Important Metric in Spam Detection:")
print("-> Precision (to avoid marking real emails as spam)")

print("\n5. Model Evaluation:")
print("Accuracy is not reliable due to class imbalance.")
print("Confusion matrix provides detailed insights.")

print("\n6. Most Reliable Metric:")
print("F1 Score (balances precision and recall)")

print("\n7. Suggested Improvements:")
print("- Handle class imbalance (SMOTE / resampling)")
print("- Tune model thresholds")
print("- Use better classification models")