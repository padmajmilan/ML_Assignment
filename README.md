# Bank Marketing Classification - ML Assignment 2

## a. Problem Statement

The goal of this project is to build a machine learning classification system to predict whether a client of a Portuguese banking institution will subscribe to a term deposit based on direct marketing campaign data. This is a **binary classification problem** where the target variable `y` indicates whether the client subscribed ("yes") or not ("no").

The business objective is to help the bank identify potential customers who are more likely to subscribe to term deposits, enabling more efficient and targeted marketing campaigns. By accurately predicting customer behavior, the bank can:
- Reduce marketing costs by focusing on high-potential customers
- Improve conversion rates for marketing campaigns
- Optimize resource allocation for customer outreach

We implement and compare **6 different machine learning models** with hyperparameter tuning to find the best performing model for this classification task.

---

## b. Dataset Description

**Dataset:** UCI Bank Marketing Dataset

**Source:** UCI Machine Learning Repository - Bank Marketing Data Set

**Total Records:** 41,873 records (with 8,375 records in test data - 20% of total records)

**Target Variable:** `y` - Whether the client subscribed to a term deposit (yes/no)

**Target Distribution:** Imbalanced dataset (~88% No, ~12% Yes)

## 🖥️ Streamlit App Features

1. **🏠 Home** - Overview of the dataset and models
2. **📊 Validate Models** - Compare all models with visualizations of its evaluation metrics once you upload the test data.There is a provision "Download Test Data" to download the test data.

---

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Client's age in years |
| `job` | Categorical | Type of job (admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown) |
| `marital` | Categorical | Marital status (divorced, married, single) |
| `education` | Categorical | Education level (primary, secondary, tertiary, unknown) |
| `default` | Binary | Has credit in default? (yes/no) |
| `balance` | Numeric | Average yearly balance in euros |
| `housing` | Binary | Has housing loan? (yes/no) |
| `loan` | Binary | Has personal loan? (yes/no) |
| `contact` | Categorical | Contact communication type (cellular, telephone, unknown) |
| `day` | Numeric | Last contact day of the month |
| `month` | Categorical | Last contact month of the year |
| `duration` | Numeric | Last contact duration in seconds |
| `campaign` | Numeric | Number of contacts during this campaign |
| `pdays` | Numeric | Days since last contact from previous campaign (-1 means not contacted) |
| `previous` | Numeric | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous marketing campaign |

### Feature Engineering (8 Custom Features)

| Feature | Description |
|---------|-------------|
| `age_group` | Age categorized into Young (<30), Mid (30-45), Senior (45-60), Old (>60) |
| `contact_efficiency` | Ratio of call duration to number of campaign contacts |
| `had_previous_contact` | Binary indicator for prior contact (1 if contacted before, 0 otherwise) |
| `balance_category` | Balance grouped into Negative, Low, Medium, High, Very High |
| `log_duration` | Log-transformed duration to reduce skewness |
| `recently_contacted` | Flag if contacted within last 30 days |
| `job_risk` | Job categorized by stability (Low, Medium, High risk) |
| `total_contacts` | Sum of current and previous campaign contacts |

---

## c. Models Used

### Model Comparison Table - Evaluation Metrics

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|---------------------| -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.7686   | 0.8864 | 0.2487    | 0.8462 | 0.3844   | 0.3739 |
| Decision Tree       | 0.8510   | 0.8187 | 0.3327    | 0.7413 | 0.4593   | 0.4290 |
| KNN                 | 0.9126   | 0.7702 | 0.4799    | 0.2839 | 0.3568   | 0.3256 |
| Naive Bayes         | 0.8873   | 0.8668 | 0.3917    | 0.5790 | 0.4673   | 0.4166 |
| Random Forest       | 0.9175   | 0.9215 | 0.5151    | 0.5734 | 0.5427   | 0.4984 |
| XGBoost             | 0.9267   | 0.9357 | 0.6091    | 0.3944 | 0.4788   | 0.4534 |

### Hyperparameters Tuned

| Model | Hyperparameters Tuned | Best Parameters |
|-------|----------------------|-----------------|
| Logistic Regression | C, solver, class_weight | C=0.01, class_weight='balanced', solver='lbfgs' |
| Decision Tree | max_depth, min_samples_split, min_samples_leaf, class_weight | max_depth=10, min_samples_leaf=10, class_weight='balanced' |
| KNN | n_neighbors, weights, metric | n_neighbors=3, metric='manhattan', weights='uniform' |
| Naive Bayes | var_smoothing | var_smoothing=1e-09 |
| Random Forest | n_estimators, max_depth, min_samples_split, class_weight | n_estimators=100, max_depth=20, class_weight='balanced' |
| XGBoost | n_estimators, max_depth, learning_rate, subsample | n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8 |

---

## Model Performance Observations

| ML Model Name           | Observation about Model Performance                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression** | Achieves the **highest recall (0.8462)** among all models, meaning it correctly identifies nearly **85% of actual subscribers**. However, it has **very low precision (0.2487)**, resulting in a large number of false positives. This makes it suitable when the cost of missing a potential subscriber is high. As a baseline linear model, it remains highly interpretable. An **AUC of 0.8864** indicates good ranking capability despite lower overall accuracy.                      |
| **Decision Tree**       | Provides a **balanced trade-off between recall (0.7413) and precision (0.3327)**, performing better than Logistic Regression in terms of F1 score (0.4593). The model captures non-linear patterns and benefits from proper depth constraints. An **AUC of 0.8187** suggests moderate discriminative power. While interpretable, the model is prone to overfitting without careful hyperparameter tuning.                                                                                  |
| **KNN**                 | Exhibits **high accuracy (0.9126)** but **low recall (0.2839)** and **low AUC (0.7702)**, indicating strong bias toward the majority class (non-subscribers). Although precision (0.4799) is relatively good, the model fails to identify many actual subscribers. KNN struggles with class imbalance and high-dimensional feature space, making it less suitable for this problem despite its high accuracy.                                                                              |
| **Naive Bayes**         | Demonstrates **moderate and stable performance** across metrics. With a recall of **0.5790**, it outperforms KNN in identifying subscribers, though it lags behind Logistic Regression and Decision Tree. The **F1 score (0.4673)** reflects reasonable balance between precision and recall. The assumption of feature independence may limit performance. Fast training and inference make it computationally efficient, but **MCC (0.4166)** indicates moderate predictive reliability. |
| **Random Forest**       | **Best overall performing model** with the **highest F1 score (0.5427)** and **highest MCC (0.4984)**, indicating the strongest balance between precision and recall. A **high AUC (0.9215)** demonstrates excellent ranking ability. The ensemble effectively reduces overfitting and captures complex feature interactions. With recall (0.5734) and precision (0.5151) both reasonably high, this model is **well-suited for real-world deployment**.                                   |
| **XGBoost **            | Achieves the **highest accuracy (0.9267)** and **highest AUC (0.9357)**, indicating superior discriminative performance. It also has the **highest precision (0.6091)**, resulting in fewer false positives. However, recall (0.3944) is lower than Random Forest, meaning more actual subscribers are missed. This model is ideal when minimizing false positives is critical. Gradient boosting effectively captures complex non-linear patterns in the data.                            |

---

## Key Findings and Recommendations

Best Model Selection

For balanced performance:
Random Forest is the most suitable model, achieving the highest F1 score (0.5427) and highest MCC (0.4984), indicating the best balance between precision and recall on the imbalanced dataset.

For ranking and scoring customers:
XGBoost demonstrates superior discriminative ability with the highest AUC (0.9357), making it ideal for prioritizing customers based on subscription likelihood.

For maximizing subscriber detection:
Logistic Regression achieves the highest recall (0.8462), successfully identifying the majority of actual subscribers. This model is effective when the cost of missing potential subscribers is high.

For minimizing false positives:
XGBoost provides the highest precision (0.6091), resulting in fewer incorrect positive predictions, which is valuable when outreach costs are high.


## 🚀 To run the app locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

```bash
python model/train_models.py
```
### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---


## 📦 Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- xgboost

---


## 📝 Conclusion

This project successfully implemented and evaluated six machine learning classification models for predicting customer subscription to term deposits. Among the evaluated models, the Random Forest classifier emerged as the best overall performer, achieving the highest F1 score (0.5427) and Matthews Correlation Coefficient (0.4984), indicating a strong balance between precision and recall on this imbalanced dataset. The XGBoost model demonstrated the highest accuracy (0.9267) and highest AUC (0.9357), reflecting superior discriminative and ranking capability, though with comparatively lower recall.
Overall, the study demonstrates the importance of selecting models based on task-specific objectives, where Random Forest is suitable for balanced decision-making, while XGBoost is preferred when minimizing false positives is critical.
