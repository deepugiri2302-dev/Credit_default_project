📄 Credit Risk Prediction and SHAP-Based Interpretability Report
1. Project Overview
This project develops a machine learning pipeline to predict the likelihood of loan default using structured customer data. The primary objective is to build a high-performing, interpretable model that supports transparent decision-making in financial services. The pipeline leverages LightGBM for classification and SHAP for model interpretability.

2. Data Preprocessing
- Missing Values: Numeric features were imputed using median values to handle missing data robustly.
- Categorical Encoding: Ordinal encoding was applied to categorical variables to maintain ordinal relationships and ensure compatibility with LightGBM.
- Artifacts Saved: Preprocessing steps were serialized using joblib for reproducibility.

3. Model Training and Tuning
- Model: LightGBM Classifier (objective='binary')
- Hyperparameter Tuning: RandomizedSearchCV with 40 iterations over a defined parameter grid
- Scoring Metric: ROC AUC (to balance sensitivity and specificity)
- Cross-Validation: 3-fold CV
- Best Model: Saved as best_model.pkl and best_lgb_model.joblib

4. Model Evaluation
Metric

Value

AUC:0.8731

F1-score: 0.7642

Accuracy: 0.8125

Precision: 0.7456

Recall: 0.7843

- Confusion Matrix: Saved as confusion_matrix.png
- Metrics Exported: metrics.json

5. Feature Importance
🔹 Traditional (Gain-Based)
Top 10 features by LightGBM’s gain importance:
- loan_amount
- credit_score
- income
- term
- employment_length
- dti
- purpose
- home_ownership
- annual_income
- open_accounts
Saved as: model_feature_importance.csv

6. SHAP Global Interpretability
- Explainer: TreeExplainer (optimized for LightGBM)
- Plots:
- Beeswarm: shap_summary_beeswarm.png
- Bar (mean SHAP): shap_summary_bar.png
- Top 10 SHAP Features:
- credit_score
- loan_amount
- dti
- employment_length
- term
- home_ownership
- annual_income
- purpose
- open_accounts
- delinquencies
Saved as: shap_top10.csv, shap_mean_abs.csv

7. SHAP Local Interpretability
Three diverse loan cases were selected:
- Likely Default: Highest predicted probability
- Likely Safe: Lowest predicted probability
- Borderline: Closest to 0.5
For each case:
- Waterfall Plot: waterfall_index_*.png
- Textual Explanation: local_explanation_index_*.txt
These explain which features pushed the prediction toward or away from default, with actual values and SHAP contributions.

8. Summary Report
A consolidated report was saved as quick_report.json, including:
- Performance metrics
- Top 10 SHAP features
- Top 10 gain-based features

  

9. Conclusion
This pipeline delivers a high-performing and interpretable credit risk model. SHAP explanations provide transparency at both global and individual levels, supporting ethical and data-driven lending decisions. The modular structure allows for easy retraining, auditing, and deployment.
