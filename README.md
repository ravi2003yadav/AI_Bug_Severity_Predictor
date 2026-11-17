# 🐞 AI-Based Software Bug Severity Predictor

An NLP and Machine Learning application that predicts the severity of software bug reports using their short descriptions.

## 📌 Project Overview

The AI-Based Software Bug Severity Predictor analyzes a software bug description and classifies it into one of six severity categories.

The project uses Natural Language Processing (NLP), TF-IDF text vectorization, and a Balanced Linear Support Vector Machine (SVM).

A Streamlit web application provides an interface for entering bug descriptions, viewing predictions, and examining the mathematical feature contributions behind the model's decision.

## 🎯 Objectives

- Automate the initial classification of software bug reports.
- Apply NLP techniques to process bug descriptions.
- Train and evaluate machine-learning classification models.
- Address class imbalance using balanced class weights.
- Provide an interactive web interface for predictions.
- Display feature contributions to help users examine model decisions.

## 🧠 Severity Categories

The model predicts one of the following six categories:

| Category | Description |
|---|---|
| Blocker | A potentially release-blocking issue |
| Critical | A highly severe software issue |
| Major | A significant functional issue |
| Minor | A comparatively less severe issue |
| Normal | A report classified as normal |
| Trivial | A very low-severity issue |

These labels follow the categories in the dataset.

## 📊 Dataset

**Dataset:** Bugzilla Eclipse Bug Reports Dataset

**Source:** [Hugging Face Dataset](https://huggingface.co/datasets/AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset)

Dataset information:

- Total records: 88,682
- Features: Project, Bug ID, Severity Label, Resolution Status, Short Description
- Prediction text: Short Description
- Target variable: Severity Label
- Number of severity classes: 6

The dataset contains short bug descriptions rather than complete bug reports. Therefore, the current model is designed to predict severity from short descriptions.

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Linear Support Vector Machine (Linear SVM)
- Joblib
- Streamlit

## 🔄 Machine Learning Workflow

1. Dataset collection
2. Dataset inspection and analysis
3. Text cleaning and preprocessing
4. Stratified training, validation, and test splitting
5. TF-IDF feature extraction
6. Training Logistic Regression and Linear SVM models
7. Comparing balanced and unweighted class configurations
8. Hyperparameter tuning of Linear SVM
9. Final model training and testing
10. Saving the trained model and vectorizer
11. Integrating the model into a Streamlit application

### Final Model Configuration

- Algorithm: Linear Support Vector Machine
- Class weighting: Balanced
- C parameter: 0.1
- Maximum iterations: 5,000
- TF-IDF maximum features: 5,000
- N-gram range: (1, 2)

## 📈 Final Model Performance

The selected model was evaluated on a held-out test dataset containing 13,303 bug reports.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 78.56% |
| Macro F1 Score | 0.3787 |
| Weighted F1 Score | 0.7813 |

### Class-wise F1 Scores

| Severity | F1 Score |
|---|---:|
| Blocker | 0.24 |
| Critical | 0.69 |
| Major | 0.17 |
| Minor | 0.10 |
| Normal | 0.88 |
| Trivial | 0.20 |

**Important:** The dataset is highly imbalanced. The model performs better on Normal and Critical reports than on several less frequent categories. Accuracy alone does not represent its performance across all severity classes.

## ✨ Application Features

- Enter a software bug description.
- Predict one of six severity categories.
- Display results in a color-coded prediction card.
- Maintain the latest ten predictions during the current session.
- Clear prediction history.
- View model evaluation information.
- Examine positive and negative feature contributions to the decision-score difference between the predicted class and runner-up.

The feature contributions are mathematical explanations of the Linear SVM decision scores. They are not prediction probabilities or proof of causation.

## 💻 Project Structure

```text
AI_Bug_Severity_Predictor/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── bugzilla_reports.csv
│   │
│   └── processed/
│       ├── cleaned_bug_reports.csv
│       ├── final_train.csv
│       ├── validation.csv
│       └── final_test.csv
│
├── models/
│   ├── final_svm_model.joblib
│   └── final_tfidf_vectorizer.joblib
│
├── results/
│   ├── final_confusion_matrix.csv
│   └── final_classification_report.txt
│
├── src/
│   ├── create_final_split.py
│   ├── compare_validation_models.py
│   ├── tune_svm.py
│   └── evaluate_final_model.py
│
├── requirements.txt
└── README.md
```

Additional analysis and preprocessing scripts may also be present in the `src` and project root directories.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Open the project directory

```bash
cd AI_Bug_Severity_Predictor
```

### 3. Install dependencies

Make sure Python is installed, then run:

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app/app.py
```

### 5. Open the application

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 📦 Dependencies

The project uses these pinned package versions:

```text
streamlit==1.67.0
scikit-learn==1.9.1
pandas==3.1.1
joblib==1.6.0
```

The saved model and vectorizer were created using the specified scikit-learn version. Use a compatible environment when loading these serialized objects.

## ⚠️ Limitations

- The model uses short bug descriptions, not complete bug reports.
- The dataset contains a substantial imbalance between severity classes.
- Performance on Major, Minor, Blocker, and Trivial categories is limited.
- Predictions are estimates and should be reviewed by software developers.
- The application does not replace manual software testing or developer judgment.
- The model's feature contributions describe its decision scores, not causal explanations.

## 🔮 Future Improvements

- Improve classification performance for underrepresented severity categories.
- Investigate additional text preprocessing techniques.
- Experiment with further machine-learning models.
- Explore additional bug-report features.
- Improve the application's reporting and visualization capabilities.
- Deploy the application online.

## 👨‍💻 Project Information

**Project Title:** AI-Based Software Bug Severity Predictor

**Domain:** Artificial Intelligence and Machine Learning

**Application Framework:** Streamlit

**Model:** Balanced Linear SVM

**Version:** 1.0

---

*Developed as an academic machine-learning project.*