
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# STEP 1: Load datasets

train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")

X_train = train_df["cleaned_description"]
y_train = train_df["Severity Label"]

X_test = test_df["cleaned_description"]
y_test = test_df["Severity Label"]

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

# STEP 3: Create two Logistic Regression models

models = {
    "Balanced": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Unweighted": LogisticRegression(
        class_weight=None,
        max_iter=1000,
        random_state=42
    )
}

# STEP 4: Train and evaluate both models

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

# STEP 5: Compare results

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

print("\nComparison completed!")
