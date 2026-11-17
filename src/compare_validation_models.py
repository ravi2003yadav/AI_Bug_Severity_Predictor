
# ==========================================================
# AI-BASED SOFTWARE BUG SEVERITY PREDICTOR
# VALIDATION-BASED MODEL COMPARISON
# ==========================================================


# STEP 1: IMPORT REQUIRED LIBRARIES

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# STEP 2: LOAD TRAINING AND VALIDATION DATASETS

train_df = pd.read_csv(
    "data/processed/final_train.csv"
)

validation_df = pd.read_csv(
    "data/processed/validation.csv"
)

print("Datasets loaded successfully!")

print("Training dataset:", train_df.shape)

print("Validation dataset:", validation_df.shape)


# STEP 3: SEPARATE INPUT FEATURES AND TARGET LABELS

# Training inputs
X_train = train_df["cleaned_description"]

# Correct training severity labels
y_train = train_df["Severity Label"]

# Validation inputs
X_validation = validation_df["cleaned_description"]

# Correct validation severity labels
y_validation = validation_df["Severity Label"]


# STEP 4: CONVERT TEXT INTO NUMERICAL FEATURES USING TF-IDF

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Learn vocabulary from training data
# and convert training text into numerical features

X_train_tfidf = tfidf.fit_transform(X_train)

# Convert validation text using the same vocabulary

X_validation_tfidf = tfidf.transform(X_validation)


print("\nTF-IDF conversion completed!")

print("Training features:", X_train_tfidf.shape)

print("Validation features:", X_validation_tfidf.shape)


# STEP 5: CREATE FOUR MACHINE LEARNING MODELS

models = {

    # Logistic Regression with class balancing

    "Balanced Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    # Logistic Regression without class balancing

    "Unweighted Logistic Regression": LogisticRegression(
        class_weight=None,
        max_iter=1000,
        random_state=42
    ),

    # Linear SVM with class balancing

    "Balanced Linear SVM": LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    ),

    # Linear SVM without class balancing

    "Unweighted Linear SVM": LinearSVC(
        class_weight=None,
        max_iter=3000,
        random_state=42
    )

}


# STEP 6: CREATE AN EMPTY LIST TO STORE RESULTS

results = []


# STEP 7: TRAIN AND EVALUATE ALL FOUR MODELS

for name, model in models.items():

    print("\n" + "=" * 55)

    print("Training:", name)

    print("=" * 55)


    # TRAIN THE MODEL

    model.fit(
        X_train_tfidf,
        y_train
    )


    # MAKE PREDICTIONS ON VALIDATION DATA

    predictions = model.predict(
        X_validation_tfidf
    )


    # CALCULATE ACCURACY

    accuracy = accuracy_score(
        y_validation,
        predictions
    )


    # CALCULATE MACRO F1-SCORE

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro"
    )


    # STORE MODEL RESULTS

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Macro F1": macro_f1

    })


    # DISPLAY CLASSIFICATION REPORT

    print("\nCLASSIFICATION REPORT:")

    print(
        classification_report(
            y_validation,
            predictions,
            zero_division=0
        )
    )


# STEP 8: CREATE MODEL COMPARISON TABLE

results_df = pd.DataFrame(results)


# STEP 9: SORT MODELS BY MACRO F1-SCORE

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)


# STEP 10: DISPLAY FINAL COMPARISON

print("\n")

print("=" * 65)

print("VALIDATION MODEL COMPARISON")

print("=" * 65)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Macro F1": "{:.4f}".format
        }
    )
)


print("\n" + "=" * 65)

print("VALIDATION EXPERIMENTS COMPLETED!")

print("=" * 65)
