 Lung Cancer Survival Prediction using Machine Learning
 
Project Overview

This project predicts patient survival outcomes for lung cancer based on demographic, clinical, and lifestyle data.
It combines supervised learning models for prediction and unsupervised clustering for patient profiling — enabling insights into treatment response and risk stratification.

 Objective

To develop a predictive model that estimates a lung cancer patient’s survival likelihood and to identify patterns that can assist in designing personalized treatment strategies.

 Dataset

Size: 890,000 entries × 17 features

Sample Columns:
age, gender, family_history, bmi, cholesterol_level, smoking_status, cancer_stage, treatment_type, survived

Sampling: Representative sampling (10%) used for performance optimization.

 Tech Stack

Language: Python

Libraries:
NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, imblearn, PCA, KMeans

Environment: Jupyter Notebook

 Methodology
 1. Data Preprocessing
Handled categorical variables using one-hot encoding.
Normalized numerical features using StandardScaler.
Balanced class distribution with SMOTE due to survival imbalance.

Logistic Regression → Accuracy: 0.52 | F1: 0.30 | AUC: 0.51
Random Forest → Accuracy: 0.69 | F1: 0.20 | AUC: 0.50
Stacking Classifier → Accuracy: 0.70 | F1: 0.18 | AUC: 0.50
Voting Classifier → Accuracy: 0.67 | F1: 0.22 | AUC: 0.50

Random Forest achieved the highest accuracy, while Logistic Regression generalized better.

3. Unsupervised Learning (K-Means + PCA)

Clustered patients into groups (k=3,4,5) using K-Means.

PCA visualizations showed distinct patient clusters:

Cluster 0: Younger patients, high BMI & cholesterol

Cluster 1: Elderly, high comorbidities → high-risk group

Cluster 2: Healthier individuals with lower BMI & cholesterol

4. Feature Importance (Random Forest)

Age → 0.259
BMI → 0.256
Cholesterol Level → 0.255

 Key Insights

Survival prediction is influenced most by age, BMI, and cholesterol.

The model struggles with minority class prediction — highlighting the challenge of medical data imbalance.

Clustering analysis supports personalized treatment grouping.

 Interpretation of Errors

False Positives: Predict survival when the patient doesn’t — may lead to under-treatment and missed interventions.

False Negatives: Predict non-survival when the patient survives — may lead to unnecessary aggressive treatment (e.g., chemotherapy).

Conclusion

Random Forest yielded the best accuracy but showed mild overfitting.

Logistic Regression offered more generalizable predictions.

PCA and clustering helped identify patient subgroups for targeted therapies.

SMOTE improved balance but not overall predictive power — future work could include deep learning or ensemble tuning.
