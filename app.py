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
    ["🏠 Home", "📊 Model Comparison", "✅ Validate Models", "🔍 Make Predictions", "📈 Feature Importance"]
)

# Check if models exist
models_exist = os.path.exists("model/model_metrics.csv")


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


# ==================== MODEL COMPARISON PAGE ====================
elif page == "📊 Model Comparison":
    st.header("📊 Model Performance Comparison (Training Data)")

    if not models_exist:
        st.error("❌ Models not trained yet. Please run `python model/train_models.py` first.")
    else:
        metrics_df = pd.read_csv("model/model_metrics.csv")

        st.subheader("📋 Evaluation Metrics")
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]),
            use_container_width=True
        )

        best_f1_model = metrics_df.loc[metrics_df["F1"].idxmax(), "Model"]
        best_auc_model = metrics_df.loc[metrics_df["AUC"].idxmax(), "Model"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best F1 Score", f"{metrics_df['F1'].max():.4f}", best_f1_model)
        with col2:
            st.metric("Best AUC", f"{metrics_df['AUC'].max():.4f}", best_auc_model)
        with col3:
            st.metric("Best Accuracy", f"{metrics_df['Accuracy'].max():.4f}",
                      metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"])

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Metrics Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_melted = metrics_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
            sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric", ax=ax)
            plt.xticks(rotation=45, ha="right")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("🎯 F1 Score Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette("viridis", len(metrics_df))
            bars = ax.barh(metrics_df["Model"], metrics_df["F1"], color=colors)
            ax.set_xlabel("F1 Score")
            for bar, val in zip(bars, metrics_df["F1"]):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')
            plt.tight_layout()
            st.pyplot(fig)

        if os.path.exists("model/best_params.csv"):
            st.markdown("---")
            st.subheader("⚙️ Best Hyperparameters")
            params_df = pd.read_csv("model/best_params.csv")
            st.dataframe(params_df, use_container_width=True)


# ==================== VALIDATE MODELS PAGE ====================
elif page == "✅ Validate Models":
    st.header("✅ Validate Models with Validation Dataset")

    if not models_exist:
        st.error("❌ Models not trained yet. Please run `python model/train_models.py` first.")
    else:
        st.markdown("""
        This page validates trained models using the **validation dataset** (3,639 records held out from training).
        
        **📁 Files:**
        - `model/validation_data_for_prediction.csv` - **Without** target column (for making predictions)
        - `model/validation_data_with_labels.csv` - **With** target column (for evaluation only)
        """)

        validation_exists = os.path.exists("model/validation_data_with_labels.csv")

        if not validation_exists:
            st.warning("⚠️ Validation data not prepared. Please run `python prepare_validation.py` first.")
        else:
            model_names = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]

            tab1, tab2 = st.tabs(["📊 Single Model Validation", "📈 Compare All Models"])

            with tab1:
                model_name = st.selectbox("Select Model to Validate", model_names)

                if st.button("🔄 Validate Model", type="primary"):
                    with st.spinner(f"Validating {model_name}..."):
                        df = pd.read_csv("model/validation_data_with_labels.csv", sep=";")
                        df = apply_feature_engineering(df)

                        y_encoder = LabelEncoder()
                        y = y_encoder.fit_transform(df["y"])
                        df = encode_features(df, exclude_cols=["y"])
                        X = df.drop("y", axis=1)

                        model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")
                        scaler = joblib.load("model/scaler.pkl")

                        if model_name in ["KNN", "Logistic Regression"]:
                            X_pred = scaler.transform(X)
                        else:
                            X_pred = X

                        y_pred = model.predict(X_pred)
                        y_prob = model.predict_proba(X_pred)[:, 1]

                        st.markdown("---")
                        st.subheader(f"📊 {model_name} - Validation Results")

                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
                        col2.metric("AUC", f"{roc_auc_score(y, y_prob):.4f}")
                        col3.metric("Precision", f"{precision_score(y, y_pred):.4f}")
                        col4.metric("Recall", f"{recall_score(y, y_pred):.4f}")
                        col5.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")
                        col6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

                        st.markdown("---")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("📝 Classification Report")
                            st.text(classification_report(y, y_pred, target_names=["No", "Yes"]))

                        with col2:
                            st.subheader("🔢 Confusion Matrix")
                            cm = confusion_matrix(y, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            st.pyplot(fig)

                        st.subheader("📉 ROC Curve")
                        fpr, tpr, _ = roc_curve(y, y_prob)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(fpr, tpr, color='darkorange', lw=2,
                                label=f'ROC curve (AUC = {auc(fpr, tpr):.4f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title(f'ROC Curve - {model_name}')
                        ax.legend(loc="lower right")
                        plt.tight_layout()
                        st.pyplot(fig)

            with tab2:
                st.subheader("📈 Compare All Models on Validation Data")

                if st.button("🔄 Validate All Models", type="primary"):
                    with st.spinner("Validating all models..."):
                        df_original = pd.read_csv("model/validation_data_with_labels.csv", sep=";")

                        validation_results = []
                        roc_curves = {}

                        for m_name in model_names:
                            df = df_original.copy()
                            df = apply_feature_engineering(df)

                            y_encoder = LabelEncoder()
                            y = y_encoder.fit_transform(df["y"])
                            df = encode_features(df, exclude_cols=["y"])
                            X = df.drop("y", axis=1)

                            model = joblib.load(f"model/{m_name.replace(' ', '_')}.pkl")
                            scaler = joblib.load("model/scaler.pkl")

                            if m_name in ["KNN", "Logistic Regression"]:
                                X_pred = scaler.transform(X)
                            else:
                                X_pred = X

                            y_pred = model.predict(X_pred)
                            y_prob = model.predict_proba(X_pred)[:, 1]

                            validation_results.append({
                                "Model": m_name,
                                "Accuracy": round(accuracy_score(y, y_pred), 4),
                                "AUC": round(roc_auc_score(y, y_prob), 4),
                                "Precision": round(precision_score(y, y_pred), 4),
                                "Recall": round(recall_score(y, y_pred), 4),
                                "F1": round(f1_score(y, y_pred), 4),
                                "MCC": round(matthews_corrcoef(y, y_pred), 4)
                            })

                            fpr, tpr, _ = roc_curve(y, y_prob)
                            roc_curves[m_name] = (fpr, tpr, roc_auc_score(y, y_prob))

                        results_df = pd.DataFrame(validation_results)
                        st.subheader("📋 Validation Metrics Comparison")
                        st.dataframe(
                            results_df.style.highlight_max(axis=0, subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]),
                            use_container_width=True
                        )

                        best_model = results_df.loc[results_df["F1"].idxmax(), "Model"]
                        best_f1 = results_df["F1"].max()
                        st.success(f"🏆 **Best Model on Validation Data:** {best_model} (F1 = {best_f1:.4f})")

                        st.markdown("---")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("📊 F1 Score Comparison")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = sns.color_palette("viridis", len(results_df))
                            bars = ax.barh(results_df["Model"], results_df["F1"], color=colors)
                            ax.set_xlabel("F1 Score")
                            for bar, val in zip(bars, results_df["F1"]):
                                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')
                            plt.tight_layout()
                            st.pyplot(fig)

                        with col2:
                            st.subheader("📉 ROC Curves Comparison")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = sns.color_palette("husl", len(roc_curves))
                            for (m_name, (fpr, tpr, auc_val)), color in zip(roc_curves.items(), colors):
                                ax.plot(fpr, tpr, color=color, lw=2, label=f'{m_name} (AUC = {auc_val:.4f})')
                            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.legend(loc="lower right", fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig)

                        results_df.to_csv("model/validation_metrics.csv", index=False)
                        st.info("💾 Validation metrics saved to `model/validation_metrics.csv`")


# ==================== MAKE PREDICTIONS PAGE ====================
elif page == "🔍 Make Predictions":
    st.header("🔍 Make Predictions")

    # ==================== DOWNLOAD TEST DATA ====================
    test_data_path = "model/validation_data_for_prediction.csv"

    if os.path.exists(test_data_path):
        with open(test_data_path, "rb") as file:
            st.download_button(
                label="📥 Download Test Data",
                data=file,
                file_name="model/validation_data_for_prediction.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ Test data file not found: model/validation_data_for_prediction.csv")

    st.markdown("---")


    if not models_exist:
        st.error("❌ Models not trained yet. Please run `python model/train_models.py` first.")
    else:
        model_name = st.selectbox(
            "Select Model for Prediction",
            ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
        )

        st.markdown("---")

        st.info("💡 **Tip:** Upload `model/validation_data_for_prediction.csv` to test predictions on validation data (without target column).")

        tab1, tab2 = st.tabs(["📁 Upload CSV File", "✍️ Manual Input"])

        with tab1:
            uploaded_file = st.file_uploader("Upload CSV File (without target column 'y')", type=["csv"])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, sep=";")
                    st.write("**Uploaded Data Preview:**")
                    st.dataframe(df.head())

                    has_target = "y" in df.columns

                    df = apply_feature_engineering(df)

                    if has_target:
                        y_encoder = LabelEncoder()
                        y = y_encoder.fit_transform(df["y"])
                        df = encode_features(df, exclude_cols=["y"])
                        X = df.drop("y", axis=1)
                    else:
                        df = encode_features(df)
                        X = df

                    model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")
                    scaler = joblib.load("model/scaler.pkl")

                    if model_name in ["KNN", "Logistic Regression"]:
                        X_pred = scaler.transform(X)
                    else:
                        X_pred = X

                    y_pred = model.predict(X_pred)
                    y_prob = model.predict_proba(X_pred)[:, 1]

                    st.markdown("---")
                    st.subheader("📊 Prediction Results")

                    results_df = pd.DataFrame({
                        "Prediction": ["Yes" if p == 1 else "No" for p in y_pred],
                        "Probability": [f"{p:.2%}" for p in y_prob]
                    })
                    st.dataframe(results_df.head(20))

                    # Summary
                    st.markdown(f"**Total Predictions:** {len(y_pred)}")
                    st.markdown(f"**Predicted Yes:** {sum(y_pred == 1)} ({sum(y_pred == 1)/len(y_pred)*100:.1f}%)")
                    st.markdown(f"**Predicted No:** {sum(y_pred == 0)} ({sum(y_pred == 0)/len(y_pred)*100:.1f}%)")

                    # Download predictions
                    download_df = pd.DataFrame({
                        "Prediction": ["Yes" if p == 1 else "No" for p in y_pred],
                        "Probability": y_prob
                    })
                    csv = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                    if has_target:
                        st.markdown("---")
                        st.subheader("📈 Model Performance (data had target column)")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.text("Classification Report:")
                            st.text(classification_report(y, y_pred, target_names=["No", "Yes"]))

                        with col2:
                            st.text("Confusion Matrix:")
                            cm = confusion_matrix(y, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        with tab2:
            st.subheader("Enter Client Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid",
                                           "management", "retired", "self-employed", "services",
                                           "student", "technician", "unemployed", "unknown"])
                marital = st.selectbox("Marital Status", ["divorced", "married", "single"])
                education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])

            with col2:
                default = st.selectbox("Has Credit Default?", ["no", "yes"])
                balance = st.number_input("Account Balance", min_value=-10000, max_value=100000, value=1000)
                housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
                loan = st.selectbox("Has Personal Loan?", ["no", "yes"])

            with col3:
                contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
                day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
                month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun",
                                               "jul", "aug", "sep", "oct", "nov", "dec"])
                duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=200)

            col4, col5 = st.columns(2)

            with col4:
                campaign = st.number_input("Number of Contacts (this campaign)", min_value=1, max_value=50, value=1)
                pdays = st.number_input("Days Since Last Contact (-1 if never)", min_value=-1, max_value=999, value=-1)

            with col5:
                previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=0)
                poutcome = st.selectbox("Previous Outcome", ["failure", "other", "success", "unknown"])

            if st.button("🔮 Predict", type="primary"):
                input_data = pd.DataFrame({
                    "age": [age], "job": [job], "marital": [marital], "education": [education],
                    "default": [default], "balance": [balance], "housing": [housing], "loan": [loan],
                    "contact": [contact], "day": [day], "month": [month], "duration": [duration],
                    "campaign": [campaign], "pdays": [pdays], "previous": [previous], "poutcome": [poutcome]
                })

                input_data = apply_feature_engineering(input_data)
                input_data = encode_features(input_data)

                model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")
                scaler = joblib.load("model/scaler.pkl")

                if model_name in ["KNN", "Logistic Regression"]:
                    X_pred = scaler.transform(input_data)
                else:
                    X_pred = input_data

                prediction = model.predict(X_pred)[0]
                probability = model.predict_proba(X_pred)[0][1]

                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    if prediction == 1:
                        st.success(f"✅ **Prediction: YES** - Client is likely to subscribe!")
                    else:
                        st.error(f"❌ **Prediction: NO** - Client is unlikely to subscribe.")

                with col2:
                    st.metric("Subscription Probability", f"{probability:.2%}")


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
