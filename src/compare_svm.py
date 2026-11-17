
# ==========================================
# LINEAR SVM MODEL COMPARISON
# AI-Based Software Bug Severity Predictor
# ==========================================

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# STEP 1: Load datasets

train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")

X_train = train_df["cleaned_description"]
y_train = train_df["Severity Label"]

X_test = test_df["cleaned_description"]
y_test = test_df["Severity Label"]

print("Datasets loaded successfully!")


# STEP 2: Convert text into TF-IDF features

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


# STEP 3: Create Linear SVM models

models = {
    "Balanced SVM": LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    ),

    "Unweighted SVM": LinearSVC(
        class_weight=None,
        max_iter=3000,
        random_state=42
    )
}


# STEP 4: Train and evaluate models

results = []

for name, model in models.items():

    print("\nTraining", name, "model...")

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    })

    print("\n", name, "CLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


# STEP 5: Compare model performance

results_df = pd.DataFrame(results)

print("\n========== LINEAR SVM COMPARISON ==========")

print(results_df.to_string(index=False))

print("\nLinear SVM experiments completed!")
