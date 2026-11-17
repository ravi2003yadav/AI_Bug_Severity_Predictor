
import re
from pathlib import Path
from textwrap import dedent

import joblib
import streamlit as st


# ============================================================
# PART A: STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bug Severity AI",
    page_icon="🐞",
    layout="wide"
)


# ============================================================
# PART B: CUSTOM DASHBOARD DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F5F7FB;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #14213D, #263F68);
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #DBEAFE;
        font-size: 16px;
        margin-bottom: 0;
    }

    .result-card {
        background: white;
        padding: 25px;
        border-radius: 14px;
        border-left: 7px solid var(--accent);
        box-shadow: 0 3px 15px rgba(0,0,0,0.06);
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .result-label {
        color: #64748B;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        color: var(--accent);
        font-size: 30px;
        font-weight: 800;
        margin-top: 7px;
    }

    .section-heading {
        color: #172554;
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
    }

    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button {
        background-color: #2457A6;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        min-height: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PART C: FIND AND LOAD TRAINED MODEL FILES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_svm_model.joblib"
)

VECTORIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_tfidf_vectorizer.joblib"
)


@st.cache_resource
def load_artifacts():

    # Check whether the model file exists.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Check whether the vectorizer file exists.
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found: {VECTORIZER_PATH}"
        )

    # Load the trained model.
    model = joblib.load(MODEL_PATH)

    # Load the fitted TF-IDF vectorizer.
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# Load the saved objects.
try:

    model, tfidf = load_artifacts()

except Exception as error:

    st.error("Unable to load the trained model files.")

    st.exception(error)

    st.stop()


# ============================================================
# PART D: TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    # Convert all characters to lowercase.
    text = text.lower()

    # Remove spaces from the beginning and end.
    text = text.strip()

    # Replace multiple spaces with a single space.
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# PART E: SVM FEATURE CONTRIBUTION EXPLANATION
# ============================================================

def explain_prediction(text_features):

    # Get decision scores for all severity classes.
    scores = model.decision_function(text_features)[0]

    # Arrange classes from highest score to lowest score.
    ranked_indices = scores.argsort()[::-1]

    # Identify the predicted and runner-up classes.
    predicted_index = ranked_indices[0]

    runner_up_index = ranked_indices[1]

    # Convert the input TF-IDF vector into a dense array.
    feature_values = text_features.toarray().ravel()

    # Get the difference between the two class weight vectors.
    coefficient_difference = (
        model.coef_[predicted_index]
        - model.coef_[runner_up_index]
    )

    # Calculate each feature's contribution to the score difference.
    contributions = (
        feature_values * coefficient_difference
    )

    # Obtain the vocabulary learned by our TF-IDF vectorizer.
    feature_names = tfidf.get_feature_names_out()

    # Find only the features present in the entered description.
    active_indices = feature_values.nonzero()[0]

    supporting_terms = []

    opposing_terms = []

    # Examine each active word or phrase.
    for index in active_indices:

        contribution = float(contributions[index])

        term = feature_names[index]

        # Positive contribution supports the predicted class.
        if contribution > 0:

            supporting_terms.append(
                (term, contribution)
            )

        # Negative contribution supports the runner-up class.
        elif contribution < 0:

            opposing_terms.append(
                (term, contribution)
            )

    # Sort positive contributions from highest to lowest.
    supporting_terms.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # Sort negative contributions from most negative upwards.
    opposing_terms.sort(
        key=lambda item: item[1]
    )

    # Calculate the intercept difference between the two classes.
    bias_difference = float(
        model.intercept_[predicted_index]
        - model.intercept_[runner_up_index]
    )

    # Calculate the overall difference between their scores.
    score_margin = float(
        scores[predicted_index]
        - scores[runner_up_index]
    )

    # Return the five strongest terms from each group.
    return {
        "supporting_terms": supporting_terms[:5],
        "opposing_terms": opposing_terms[:5],
        "runner_up": str(model.classes_[runner_up_index]),
        "score_margin": score_margin,
        "bias_difference": bias_difference
    }


# ============================================================
# PART F: SEVERITY COLORS AND DESCRIPTIONS
# ============================================================

severity_colors = {

    "blocker": "#DC2626",

    "critical": "#EA580C",

    "major": "#CA8A04",

    "minor": "#2563EB",

    "normal": "#16A34A",

    "trivial": "#64748B"

}


severity_messages = {

    "blocker":
        "The model classified this report as Blocker.",

    "critical":
        "The model classified this report as Critical.",

    "major":
        "The model classified this report as Major.",

    "minor":
        "The model classified this report as Minor.",

    "normal":
        "The model classified this report as Normal.",

    "trivial":
        "The model classified this report as Trivial."

}




# ============================================================
# PART G: APPLICATION HEADER
# ============================================================

hero_html = (
    "<div class='hero'>"
    "<h1>🐞 Bug Severity Predictor</h1>"
    "<p>AI-powered software bug classification "
    "using Natural Language Processing and Machine Learning.</p>"
    "</div>"
)

st.markdown(
    hero_html,
    unsafe_allow_html=True
)


# ============================================================
# PART H: DASHBOARD PERFORMANCE METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        label="Test Accuracy",
        value="78.56%"
    )

with col2:

    st.metric(
        label="Macro F1 Score",
        value="0.3787"
    )

with col3:

    st.metric(
        label="Severity Categories",
        value="6"
    )


st.divider()


# ============================================================
# PART I: INPUT AREA AND MODEL INFORMATION
# ============================================================

left_column, right_column = st.columns(
    [1.7, 1],
    gap="large"
)


with left_column:

    st.markdown(
        '<div class="section-heading">'
        'Analyze a Bug Report'
        '</div>',
        unsafe_allow_html=True
    )

    # Form prevents a prediction on every text change.
    with st.form("bug_prediction_form"):

        bug_description = st.text_area(
            "Bug Description",

            placeholder=(
                "Example: Application crashes when "
                "the user clicks the login button..."
            ),

            height=180,

            max_chars=1000
        )

        submitted = st.form_submit_button(
            "Predict Bug Severity",
            type="primary",
            use_container_width=True
        )


with right_column:

    st.markdown(
        '<div class="section-heading">'
        'Model Information'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.write("**Algorithm**")

        st.write("Balanced Linear SVM")

        st.divider()

        st.write("**Text Processing**")

        st.write("TF-IDF Vectorization")

        st.divider()

        st.write("**Feature Count**")

        st.write("5,000")

        st.divider()

        st.write("**Dataset**")

        st.write("88,682 Bugzilla reports")


# ============================================================
# PART J: INITIALIZE PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ============================================================
# PART K: GENERATE PREDICTION
# ============================================================

if submitted:

    if not bug_description.strip():

        st.warning(
            "Please enter a bug description before predicting."
        )

    else:

        try:

            with st.spinner("Analyzing bug description..."):

                # Step 1: Preprocess the entered description.
                cleaned_text = preprocess_text(
                    bug_description
                )

                # Warn when input is longer than training descriptions.
                if len(cleaned_text) > 255:

                    st.warning(
                        "Your description is longer than the "
                        "255-character maximum in the training "
                        "dataset. The prediction may be less reliable."
                    )

                # Step 2: Convert text into TF-IDF features.
                text_features = tfidf.transform(
                    [cleaned_text]
                )

                # Step 3: Predict the severity category.
                prediction = model.predict(
                    text_features
                )[0]

                # Step 4: Explain the prediction using feature weights.
                explanation = explain_prediction(
                    text_features
                )

            # Step 5: Store prediction and explanation.
            prediction_record = {

                "description": bug_description,

                "prediction": str(prediction),

                "supporting_terms": (
                    explanation["supporting_terms"]
                ),

                "opposing_terms": (
                    explanation["opposing_terms"]
                ),

                "runner_up": explanation["runner_up"],

                "score_margin": explanation["score_margin"],

                "bias_difference": explanation["bias_difference"]

            }

            # Add the latest prediction at the beginning.
            st.session_state.prediction_history.insert(
                0,
                prediction_record
            )

            # Retain only the ten latest predictions.
            st.session_state.prediction_history = (
                st.session_state.prediction_history[:10]
            )

            st.success(
                "Prediction completed successfully!"
            )

        except Exception as error:

            st.error(
                "An error occurred while generating the prediction."
            )

            st.exception(error)


# ============================================================
# PART L: DISPLAY LATEST PREDICTION
# ============================================================

if st.session_state.prediction_history:

    latest_result = (
        st.session_state.prediction_history[0]
    )

    predicted_severity = (
        latest_result["prediction"]
    )

    accent = severity_colors.get(
        predicted_severity,
        "#2563EB"
    )

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        'Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )

    # Dedent removes leading spaces so HTML renders correctly.
    result_html = dedent(
        f"""
<div class="result-card" style="--accent: {accent};">
    <div class="result-label">
        Predicted Severity
    </div>
    <div class="result-value">
        {predicted_severity.upper()}
    </div>
</div>
"""
    ).strip()

    st.markdown(
        result_html,
        unsafe_allow_html=True
    )

    st.write(
        severity_messages.get(
            predicted_severity,
            "Prediction generated."
        )
    )

    st.write("**Analyzed Description:**")

    st.info(
        latest_result["description"]
    )

    st.warning(
        "This is an AI-generated estimate, not a verified "
        "developer assessment. Human review is recommended, "
        "especially for critical and high-impact bugs."
    )


    # ========================================================
    # PART M: DISPLAY FEATURE CONTRIBUTION EXPLANATION
    # ========================================================

    with st.expander(
        "Why did the model make this prediction?"
    ):

        runner_up = latest_result["runner_up"]

        st.write(
            "**Comparison:**",
            predicted_severity.upper(),
            "versus",
            runner_up.upper()
        )

        st.caption(
            "The following terms contribute to the difference "
            "between the two classes' SVM decision scores."
        )

        support_column, oppose_column = st.columns(2)

        with support_column:

            st.markdown(
                "**Terms supporting the prediction**"
            )

            supporting_terms = (
                latest_result["supporting_terms"]
            )

            if supporting_terms:

                for term, value in supporting_terms:

                    st.markdown(
                        f"- `{term}`: **+{value:.4f}**"
                    )

            else:

                st.write(
                    "No positive feature contributions found."
                )

        with oppose_column:

            st.markdown(
                "**Terms favoring the runner-up**"
            )

            opposing_terms = (
                latest_result["opposing_terms"]
            )

            if opposing_terms:

                for term, value in opposing_terms:

                    st.markdown(
                        f"- `{term}`: **{value:.4f}**"
                    )

            else:

                st.write(
                    "No negative feature contributions found."
                )

        st.divider()

        score_margin = latest_result["score_margin"]

        bias_difference = latest_result["bias_difference"]

        st.metric(
            label="Decision score difference",
            value=f"{score_margin:.4f}"
        )

        st.caption(
            "Constant intercept difference: "
            f"{bias_difference:+.4f}"
        )

        st.info(
            "Positive term contributions favor the predicted class "
            "over the runner-up. Negative contributions favor the "
            "runner-up. The intercept also affects the final score. "
            "These values are mathematical feature contributions, "
            "not probabilities, confidence percentages, or proof "
            "of causation."
        )


# ============================================================
# PART N: RECENT PREDICTION HISTORY
# ============================================================

if st.session_state.prediction_history:

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        'Recent Predictions'
        '</div>',
        unsafe_allow_html=True
    )

    for index, item in enumerate(
        st.session_state.prediction_history,
        start=1
    ):

        st.write(
            f"**{index}. {item['prediction'].upper()}**"
        )

        st.caption(
            item["description"]
        )

    if st.button("Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()


# ============================================================
# PART O: ABOUT THE MODEL AND ITS LIMITATIONS
# ============================================================

st.divider()

with st.expander(
    "About the model and its limitations"
):

    st.write(
        """
        This project uses a Balanced Linear Support Vector Machine
        trained on Bugzilla Eclipse bug reports.

        The input text is processed using TF-IDF vectorization.

        The training dataset contains 88,682 bug reports.

        The model predicts six severity categories:

        - Blocker
        - Critical
        - Major
        - Minor
        - Normal
        - Trivial

        Final test performance:

        - Accuracy: 78.56%
        - Macro F1: 0.3787

        Performance is uneven across severity categories.
        The model has particular difficulty identifying
        Major and Minor reports.

        The model was trained using short bug descriptions,
        not complete bug reports.

        Predictions and feature contributions should support
        human review, not replace developer judgment.
        """
    )



# ============================================================
# PART P: FOOTER
# ============================================================

footer_html = dedent("""
<div style="
    text-align: center;
    color: #64748B;
    font-size: 13px;
    padding-top: 25px;
">
    AI-Based Software Bug Severity Predictor
    <br>
    Project Version 1 | NLP + Machine Learning
</div>
""").strip()

st.markdown(
    footer_html,
    unsafe_allow_html=True
)
