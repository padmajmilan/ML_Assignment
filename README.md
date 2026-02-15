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

**Total Records:** 45,211 records (split into Train: 36,000 records (from 'model/bank-train-data.csv') | Test: 4,000 | Validation: 3,639)

**Target Variable:** `y` - Whether the client subscribed to a term deposit (yes/no)

**Target Distribution:** Imbalanced dataset (~88% No, ~12% Yes)

## 🖥️ Streamlit App Features

1. **🏠 Home** - Overview of the dataset and models
2. **📊 Model Comparison** - Compare all models with visualizations of its evaluation metrics using training data.
3. **✅ Validate Models** - Validate models using held-out validation data.Click "Validate Model" button.
   "Single Model Validation" gives "Classification Report" and "Confusion Matrix"
4. **🔍 Make Predictions** - Download the test data without output label using "📥 Download Test Data" button
   and use it to make predictions via "Upload CSV file" or enter data manually
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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8103 | 0.8881 | 0.3520 | 0.8015 | 0.4892 | 0.4440 |
| Decision Tree | 0.8296 | 0.8714 | 0.3803 | 0.8002 | 0.5156 | 0.4713 |
| KNN | 0.8918 | 0.7810 | 0.5351 | 0.3456 | 0.4200 | 0.3739 |
| Naive Bayes | 0.8431 | 0.8470 | 0.3758 | 0.5821 | 0.4567 | 0.3821 |
| Random Forest (Ensemble) | 0.8962 | 0.9189 | 0.5366 | 0.6201 | 0.5753 | 0.5183 |
| XGBoost (Ensemble) | 0.9096 | 0.9320 | 0.6359 | 0.4730 | 0.5425 | 0.5002 |

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

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved the **highest recall (0.8015)** among all models, meaning it correctly identifies ~80% of actual subscribers. However, it has low precision (0.3520), resulting in many false positives. Good baseline model with interpretable coefficients. The high recall makes it suitable when the cost of missing a potential subscriber is high. AUC of 0.8881 indicates good ranking ability. |
| **Decision Tree** | Shows balanced performance with **second-highest recall (0.8002)** similar to Logistic Regression. Provides interpretable decision rules but slightly lower AUC (0.8714). The model benefits from class_weight='balanced' to handle the imbalanced dataset. Prone to overfitting without proper depth constraints. F1 score of 0.5156 is moderate. |
| **KNN** | Has the **highest accuracy (0.8918)** among non-ensemble models but **lowest AUC (0.7810)** and recall (0.3456). This indicates the model is biased toward the majority class (non-subscribers). KNN struggles with the imbalanced dataset and high dimensionality. Best precision among simple models (0.5351) but misses many actual subscribers. |
| **Naive Bayes** | Moderate performance across all metrics. Recall of 0.5821 is better than KNN but worse than Logistic Regression and Decision Tree. The assumption of feature independence may not hold well for this dataset. Fast training and prediction but **lowest MCC (0.3821)** among models with decent recall, indicating weaker correlation between predictions and actual values. |
| **Random Forest (Ensemble)** | **Best overall model** with highest F1 score (0.5753) and MCC (0.5183), indicating the best balance between precision and recall. Strong AUC (0.9189) shows excellent ranking capability. Ensemble of 100 trees with max_depth=20 provides robust predictions. Good recall (0.6201) while maintaining reasonable precision (0.5366). Recommended for production deployment. |
| **XGBoost (Ensemble)** | **Highest accuracy (0.9096)** and **highest AUC (0.9320)** among all models, demonstrating superior discriminative ability. Best precision (0.6359) means fewer false positives. However, recall (0.4730) is lower than Random Forest, missing more actual subscribers. Ideal when false positives are costly. Gradient boosting effectively captures complex patterns in the data. |

---

## Key Findings and Recommendations

### Best Model Selection

1. **For balanced performance:** **Random Forest** is recommended with the highest F1 (0.5753) and MCC (0.5183)
2. **For ranking/scoring customers:** **XGBoost** has the highest AUC (0.9320)
3. **For maximizing subscriber detection:** **Logistic Regression** has the highest recall (0.8015)
4. **For minimizing false positives:** **XGBoost** has the highest precision (0.6359)

### Challenges Addressed

- **Class Imbalance:** Handled using `class_weight='balanced'` parameter and stratified sampling
- **Feature Engineering:** Created 8 domain-specific features to improve model performance
- **Hyperparameter Tuning:** Used GridSearchCV with 5-fold cross-validation for optimal parameters



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

This project successfully implemented and compared 6 machine learning models for predicting term deposit subscriptions. The **Random Forest** model emerged as the best overall performer with balanced precision-recall trade-off (F1=0.5753), while **XGBoost** achieved the highest discriminative ability (AUC=0.9320). Feature engineering and hyperparameter tuning significantly improved model performance on this imbalanced classification task.

