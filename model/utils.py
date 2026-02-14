"""
Utility functions for Bank Marketing Classification
ML Assignment 2
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def apply_feature_engineering(df):
    """
    Apply feature engineering transformations to the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with original features

    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()

    # 1. Age Group Binning
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["Young", "Mid", "Senior", "Old"]
    )

    # 2. Contact Efficiency
    df["contact_efficiency"] = df["duration"] / (df["campaign"] + 1)

    # 3. Previous Contact Indicator
    df["had_previous_contact"] = np.where(df["previous"] > 0, 1, 0)

    # 4. Balance Category
    df["balance_category"] = pd.cut(
        df["balance"],
        bins=[-np.inf, 0, 500, 2000, 10000, np.inf],
        labels=["Negative", "Low", "Medium", "High", "Very_High"]
    )

    # 5. Log Transform Duration
    df["log_duration"] = np.log1p(df["duration"])

    # 6. Recently Contacted
    df["recently_contacted"] = np.where(
        (df["pdays"] > 0) & (df["pdays"] < 30), 1, 0
    )

    # 7. Job Risk Category
    high_risk_jobs = ["unemployed", "unknown", "student"]
    medium_risk_jobs = ["self-employed", "entrepreneur", "services", "housemaid"]
    df["job_risk"] = df["job"].apply(
        lambda x: "High" if x in high_risk_jobs
        else ("Medium" if x in medium_risk_jobs else "Low")
    )

    # 8. Total Contacts
    df["total_contacts"] = df["campaign"] + df["previous"]

    return df


def encode_categorical(df, encoders=None):
    """
    Encode categorical variables using LabelEncoder.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    encoders : dict, optional
        Dictionary of pre-fitted encoders

    Returns:
    --------
    tuple
        (encoded dataframe, encoders dictionary)
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == "y":
            continue
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
        else:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder

    return df, encoders


def get_feature_descriptions():
    """
    Get descriptions of all features including engineered ones.

    Returns:
    --------
    dict
        Dictionary with feature names and descriptions
    """
    return {
        # Original features
        "age": "Client's age in years",
        "job": "Type of job",
        "marital": "Marital status",
        "education": "Education level",
        "default": "Has credit in default?",
        "balance": "Average yearly balance in euros",
        "housing": "Has housing loan?",
        "loan": "Has personal loan?",
        "contact": "Contact communication type",
        "day": "Last contact day of the month",
        "month": "Last contact month of the year",
        "duration": "Last contact duration in seconds",
        "campaign": "Number of contacts during this campaign",
        "pdays": "Days since last contact from previous campaign (-1 means not contacted)",
        "previous": "Number of contacts before this campaign",
        "poutcome": "Outcome of the previous marketing campaign",

        # Engineered features
        "age_group": "Age categorized into Young (<30), Mid (30-45), Senior (45-60), Old (>60)",
        "contact_efficiency": "Ratio of call duration to number of campaign contacts",
        "had_previous_contact": "Binary flag indicating if client was contacted before",
        "balance_category": "Balance grouped into Negative, Low, Medium, High, Very High",
        "log_duration": "Log-transformed duration to reduce skewness",
        "recently_contacted": "Flag if client was contacted within last 30 days",
        "job_risk": "Job stability category (Low, Medium, High risk)",
        "total_contacts": "Sum of current and previous campaign contacts"
    }


def print_model_summary(results_df):
    """
    Print a formatted summary of model results.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing model metrics
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)

    for _, row in results_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  Accuracy:  {row['Accuracy']:.4f}")
        print(f"  AUC:       {row['AUC']:.4f}")
        print(f"  Precision: {row['Precision']:.4f}")
        print(f"  Recall:    {row['Recall']:.4f}")
        print(f"  F1 Score:  {row['F1']:.4f}")
        print(f"  MCC:       {row['MCC']:.4f}")

    print("\n" + "="*60)
    best_model = results_df.loc[results_df['F1'].idxmax(), 'Model']
    best_f1 = results_df['F1'].max()
    print(f"BEST MODEL (by F1): {best_model} with F1 = {best_f1:.4f}")
