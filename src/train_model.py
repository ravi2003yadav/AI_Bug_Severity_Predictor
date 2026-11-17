
# ==========================================
# AI-BASED SOFTWARE BUG SEVERITY PREDICTOR
# Logistic Regression Model
# ==========================================


# STEP 1: Import required libraries

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# STEP 2: Load training and testing datasets

train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")

print("Datasets loaded successfully!")


# STEP 3: Separate input features and target labels

X_train = train_df["cleaned_description"]
y_train = train_df["Severity Label"]

X_test = test_df["cleaned_description"]
y_test = test_df["Severity Label"]


# STEP 4: Convert text into numerical features using TF-IDF

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

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)


# STEP 5: Create Logistic Regression model

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)


# STEP 6: Train the model

print("\nTraining Logistic Regression model...")

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# STEP 7: Make predictions using testing data

y_pred = model.predict(X_test_tfidf)

print("\nPredictions completed!")


# STEP 8: Evaluate model performance

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\n========== MODEL EVALUATION RESULTS ==========")

print("Accuracy:", round(accuracy, 4))

print("Macro F1-score:", round(macro_f1, 4))


# STEP 9: Display classification report

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# STEP 10: Generate confusion matrix

labels = [
    "blocker",
    "critical",
    "major",
    "minor",
    "normal",
    "trivial"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

confusion_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n========== CONFUSION MATRIX ==========")

print(confusion_df)


# STEP 11: Display predicted severity distribution

print("\n========== PREDICTED SEVERITY DISTRIBUTION ==========")

prediction_counts = pd.Series(y_pred).value_counts()

print(prediction_counts)


# STEP 12: Save trained model and TF-IDF vectorizer

joblib.dump(
    model,
    "models/logistic_regression.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("\n==========================================")
print("Model and TF-IDF vectorizer saved successfully!")
print("==========================================")
