
import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# PART A: LOAD ALL THREE DATASETS
# =====================================================

train_df = pd.read_csv("data/processed/final_train.csv")
validation_df = pd.read_csv("data/processed/validation.csv")
test_df = pd.read_csv("data/processed/final_test.csv")

print("Datasets loaded successfully!")

print("Training dataset:", train_df.shape)
print("Validation dataset:", validation_df.shape)
print("Final test dataset:", test_df.shape)


# =====================================================
# PART B: COMBINE TRAINING AND VALIDATION DATA
# =====================================================

combined_train_df = pd.concat(
    [train_df, validation_df],
    ignore_index=True
)

print("\nTraining and validation datasets combined!")
print("Combined training dataset:", combined_train_df.shape)


# =====================================================
# PART C: SEPARATE TEXT AND TARGET LABELS
# =====================================================

X_train = combined_train_df["cleaned_description"]
y_train = combined_train_df["Severity Label"]

X_test = test_df["cleaned_description"]
y_test = test_df["Severity Label"]


# =====================================================
# PART D: CONVERT TEXT INTO TF-IDF FEATURES
# =====================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF conversion completed!")

print("Training features:", X_train_tfidf.shape)
print("Testing features:", X_test_tfidf.shape)


# =====================================================
# PART E: CREATE AND TRAIN FINAL SVM MODEL
# =====================================================

final_model = LinearSVC(
    C=0.1,
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)

print("\nTraining final Balanced Linear SVM...")

final_model.fit(X_train_tfidf, y_train)

print("Final model training completed!")


# =====================================================
# PART F: MAKE PREDICTIONS ON FINAL TEST DATA
# =====================================================

y_predictions = final_model.predict(X_test_tfidf)

print("\nFinal test predictions completed!")


# =====================================================
# PART G: CALCULATE FINAL PERFORMANCE METRICS
# =====================================================

accuracy = accuracy_score(y_test, y_predictions)

macro_f1 = f1_score(
    y_test,
    y_predictions,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_predictions,
    average="weighted"
)

print("\n" + "=" * 65)
print("FINAL MODEL PERFORMANCE")
print("=" * 65)

print("Accuracy:", round(accuracy, 4))
print("Macro F1 Score:", round(macro_f1, 4))
print("Weighted F1 Score:", round(weighted_f1, 4))


# =====================================================
# PART H: GENERATE CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    y_test,
    y_predictions,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print("=" * 65)

print(report)


# =====================================================
# PART I: GENERATE CONFUSION MATRIX
# =====================================================

class_labels = [
    "blocker",
    "critical",
    "major",
    "minor",
    "normal",
    "trivial"
]

cm = confusion_matrix(
    y_test,
    y_predictions,
    labels=class_labels
)

cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels
)

print("\nCONFUSION MATRIX")
print("=" * 65)

print(cm_df)


# =====================================================
# PART J: SAVE FINAL MODEL AND RESULTS
# =====================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(
    final_model,
    "models/final_svm_model.joblib"
)

joblib.dump(
    tfidf,
    "models/final_tfidf_vectorizer.joblib"
)

cm_df.to_csv(
    "results/final_confusion_matrix.csv"
)

with open(
    "results/final_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("FINAL MODEL PERFORMANCE\n")
    file.write("=" * 65 + "\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro F1 Score: {macro_f1:.4f}\n")
    file.write(f"Weighted F1 Score: {weighted_f1:.4f}\n\n")
    file.write("CLASSIFICATION REPORT\n")
    file.write(report)
    file.write("\nCONFUSION MATRIX\n")
    file.write(cm_df.to_string())

print("\nFinal model saved successfully!")
print("Model: models/final_svm_model.joblib")
print("Vectorizer: models/final_tfidf_vectorizer.joblib")

print("\nResults saved in the results folder!")

print("\nFINAL MODEL EVALUATION COMPLETED!")
