import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
import os

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction - ML Assignment 2",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("🏦 Bank Marketing Classification")
st.markdown("### ML Assignment 2 - Term Deposit Subscription Prediction")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "✅ Validate Models"]
)

# Check if models exist
models_exist = os.path.exists("model_metrics.csv")


def apply_feature_engineering(df):
    """Apply feature engineering to dataframe"""
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 45, 60, 100],
                             labels=["Young", "Mid", "Senior", "Old"])
    df["contact_efficiency"] = df["duration"] / (df["campaign"] + 1)
    df["had_previous_contact"] = np.where(df["previous"] > 0, 1, 0)
    df["balance_category"] = pd.cut(df["balance"],
                                    bins=[-np.inf, 0, 500, 2000, 10000, np.inf],
                                    labels=["Negative", "Low", "Medium", "High", "Very_High"])
    df["log_duration"] = np.log1p(df["duration"])
    df["recently_contacted"] = np.where((df["pdays"] > 0) & (df["pdays"] < 30), 1, 0)

    high_risk_jobs = ["unemployed", "unknown", "student"]
    medium_risk_jobs = ["self-employed", "entrepreneur", "services", "housemaid"]
    df["job_risk"] = df["job"].apply(
        lambda x: "High" if x in high_risk_jobs else ("Medium" if x in medium_risk_jobs else "Low"))
    df["total_contacts"] = df["campaign"] + df["previous"]
    return df


def encode_features(df, exclude_cols=None):
    """Encode categorical features"""
    df = df.copy()
    if exclude_cols is None:
        exclude_cols = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col not in exclude_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
    return df


# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.header("Welcome to the Bank Marketing Prediction App")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 About the Dataset")
        st.markdown("""
        The **Bank Marketing Dataset** from UCI contains data from direct marketing 
        campaigns of a Portuguese banking institution. The goal is to predict whether 
        a client will subscribe to a term deposit.
        
        **Features include:**
        - **Demographics:** Age, Job, Marital Status, Education
        - **Financial:** Balance, Housing Loan, Personal Loan
        - **Campaign:** Contact Type, Duration, Number of Contacts
        - **Previous Campaign:** Days since last contact, Previous outcome
        """)

    with col2:
        st.subheader("🤖 Models Implemented")
        st.markdown("""
        Six machine learning models with hyperparameter tuning:
        
        1. **Logistic Regression** - Linear classifier with regularization
        2. **Decision Tree** - Tree-based interpretable model
        3. **K-Nearest Neighbors (KNN)** - Instance-based learning
        4. **Naive Bayes** - Probabilistic classifier
        5. **Random Forest** - Ensemble of decision trees
        6. **XGBoost** - Gradient boosting framework
        """)

    st.markdown("---")

    st.subheader("🔧 Feature Engineering Applied")
    st.markdown("""
    The following custom features were engineered to improve model performance:
    
    | Feature | Description |
    |---------|-------------|
    | `age_group` | Age categorized into Young, Mid, Senior, Old |
    | `contact_efficiency` | Ratio of call duration to campaign contacts |
    | `had_previous_contact` | Binary indicator for prior contact |
    | `balance_category` | Balance grouped into Negative, Low, Medium, High, Very High |
    | `log_duration` | Log-transformed duration to reduce skewness |
    | `recently_contacted` | Flag if contacted within last 30 days |
    | `job_risk` | Job categorized by stability (Low, Medium, High risk) |
    | `total_contacts` | Sum of current and previous campaign contacts |
    """)

    if not models_exist:
        st.warning("⚠️ Models not trained yet. Please run `python model/train_models.py` first.")

# ==================== VALIDATE MODELS PAGE ====================
elif page == "✅ Validate Models":
    st.header("✅ Validate Models using Test Data")

    if not models_exist:
        st.error("❌ Models not trained yet. Please run `python model/train_models.py` first.")
    else:
        # ---------- Download Test Data ----------
        test_data_path = "test_data.csv"

        if os.path.exists(test_data_path):
            with open(test_data_path, "rb") as f:
                st.download_button(
                    label="📥 Download Test Data",
                    data=f,
                    file_name=test_data_path,
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ Test data file not found: test_data.csv")

        st.markdown("---")

        # ---------- Upload Test Data ----------
        uploaded_file = st.file_uploader(
            "📤 Upload Test Data (CSV with target column 'y')",
            type=["csv"]
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=",")
                st.subheader("📄 Uploaded Data Preview")
                st.dataframe(df.head())

                if "y" not in df.columns:
                    st.error("❌ Uploaded file must contain target column 'y'")
                else:
                    # ---------- Preprocessing ----------
                    df = apply_feature_engineering(df)

                    y_encoder = LabelEncoder()
                    y_true = y_encoder.fit_transform(df["y"])

                    df = encode_features(df, exclude_cols=["y"])
                    X = df.drop("y", axis=1)

                    scaler = joblib.load("model/scaler.pkl")

                    model_names = [
                        "Logistic Regression",
                        "Decision Tree",
                        "KNN",
                        "Naive Bayes",
                        "Random Forest",
                        "XGBoost"
                    ]

                    results = {}
                    metrics_list = []

                    # ---------- Evaluate All Models ----------
                    with st.spinner("⏳ Evaluating all models on uploaded test data..."):
                        for model_name in model_names:
                            model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")

                            if model_name in ["Logistic Regression", "KNN"]:
                                X_pred = scaler.transform(X)
                            else:
                                X_pred = X

                            y_pred = model.predict(X_pred)
                            y_prob = model.predict_proba(X_pred)[:, 1]

                            metrics = {
                                "Model": model_name,
                                "Accuracy": accuracy_score(y_true, y_pred),
                                "AUC": roc_auc_score(y_true, y_prob),
                                "Precision": precision_score(y_true, y_pred),
                                "Recall": recall_score(y_true, y_pred),
                                "F1": f1_score(y_true, y_pred),
                                "MCC": matthews_corrcoef(y_true, y_pred)
                            }

                            metrics_list.append(metrics)
                            results[model_name] = (y_pred, y_prob)

                    metrics_df = pd.DataFrame(metrics_list)


                    st.markdown("---")
                    st.subheader("📊 Evaluation Metrics (All Models)")
                    # ---------- Best Models Summary ----------
                    best_f1_model = metrics_df.loc[metrics_df["F1"].idxmax(), "Model"]
                    best_auc_model = metrics_df.loc[metrics_df["AUC"].idxmax(), "Model"]
                    best_acc_model = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Best F1 Score",
                            f"{metrics_df['F1'].max():.4f}",
                            best_f1_model
                        )

                    with col2:
                        st.metric(
                            "Best AUC",
                            f"{metrics_df['AUC'].max():.4f}",
                            best_auc_model
                        )

                    with col3:
                        st.metric(
                            "Best Accuracy",
                            f"{metrics_df['Accuracy'].max():.4f}",
                            best_acc_model
                        )

                    st.markdown("---")

                    st.dataframe(
                        metrics_df.style.highlight_max(
                            axis=0,
                            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
                        ),
                        use_container_width=True
                    )

                    st.markdown("---")

                    # ---------- Model Selection ----------
                    selected_model = st.selectbox(
                        "🎯 Select Model for Detailed Analysis",
                        model_names
                    )

                    y_pred_sel, y_prob_sel = results[selected_model]

                    yes_pct = np.mean(y_pred_sel == 1) * 100
                    no_pct = np.mean(y_pred_sel == 0) * 100

                    st.subheader(f"📌 Prediction Summary – {selected_model}")
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted YES (%)", f"{yes_pct:.2f}%")
                    col2.metric("Predicted NO (%)", f"{no_pct:.2f}%")

                    st.markdown("---")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📝 Classification Report")
                        st.text(
                            classification_report(
                                y_true,
                                y_pred_sel,
                                target_names=["No", "Yes"]
                            )
                        )

                    with col2:
                        st.subheader("🔢 Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred_sel)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            xticklabels=["No", "Yes"],
                            yticklabels=["No", "Yes"],
                            ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error processing uploaded file: {str(e)}")




# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🏦 Bank Marketing Classification - ML Assignment 2</p>
        <p>Built with Streamlit | Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True
)
