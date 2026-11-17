
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


# =====================================================
# PART A: LOAD TRAINING AND VALIDATION DATASETS
# =====================================================

train_df = pd.read_csv("data/processed/final_train.csv")
validation_df = pd.read_csv("data/processed/validation.csv")

print("Datasets loaded successfully!")
print("Training dataset:", train_df.shape)
print("Validation dataset:", validation_df.shape)


# =====================================================
# PART B: SEPARATE INPUT TEXT AND TARGET LABELS
# =====================================================

X_train = train_df["cleaned_description"]
y_train = train_df["Severity Label"]

X_validation = validation_df["cleaned_description"]
y_validation = validation_df["Severity Label"]


# =====================================================
# PART C: CONVERT TEXT INTO TF-IDF FEATURES
# =====================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_validation_tfidf = tfidf.transform(X_validation)

print("\nTF-IDF conversion completed!")
print("Training features:", X_train_tfidf.shape)
print("Validation features:", X_validation_tfidf.shape)


# =====================================================
# PART D: DEFINE C VALUES TO EXPERIMENT WITH
# =====================================================

c_values = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]

results = []


# =====================================================
# PART E: TRAIN AND EVALUATE EACH SVM MODEL
# =====================================================

for c_value in c_values:

    print("\n" + "=" * 60)
    print("Training Balanced Linear SVM with C =", c_value)
    print("=" * 60)

    model = LinearSVC(
        C=c_value,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_validation_tfidf)

    accuracy = accuracy_score(y_validation, predictions)

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro"
    )

    results.append({
        "C": c_value,
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    })

    print("\nAccuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_validation,
            predictions,
            zero_division=0
        )
    )


# =====================================================
# PART F: DISPLAY FINAL COMPARISON
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n" + "=" * 65)
print("SVM HYPERPARAMETER TUNING RESULTS")
print("=" * 65)

print(results_df.to_string(index=False))

best_result = results_df.iloc[0]

print("\nBest C value:", best_result["C"])
print("Best Accuracy:", round(best_result["Accuracy"], 4))
print("Best Macro F1:", round(best_result["Macro F1"], 4))

print("\nSVM TUNING COMPLETED!")
