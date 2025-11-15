💳 Credit Risk Prediction with LightGBM and SHAP
📌 Project Overview
This project builds an interpretable machine learning pipeline to predict loan default risk using LightGBM. It includes robust preprocessing, hyperparameter tuning, model evaluation, and SHAP-based feature attribution to support transparent decision-making in financial services.

🧠 Objectives
- Predict binary loan default risk using gradient boosting
- Optimize model performance using randomized hyperparameter search
- Evaluate with balanced metrics (AUC, F1-score)
- Interpret predictions globally and locally using SHAP
- Compare traditional feature importance vs. SHAP explanations

🛠️ Pipeline Summary
1. Preprocessing
- Impute missing numeric values (median)
- Encode categorical features (ordinal encoding)
- Save preprocessing artifacts for reuse
2. Model Training
- LightGBM classifier with binary objective
- RandomizedSearchCV with 40 iterations
- ROC AUC used for scoring
- Best model saved as best_model.pkl
3. Evaluation
- Metrics: AUC, F1-score, accuracy, precision, recall
- Confusion matrix visualization
- Metrics saved to metrics.json
4. Interpretability
- Global SHAP summary plots (beeswarm + bar)
- Top 10 SHAP features saved to CSV
- Traditional gain-based feature importance for comparison
- Local SHAP waterfall plots for 3 diverse loan cases
- Plain-language explanations saved to .txt files

📁 Output Files
All outputs are saved in the outputs/ directory:
|  |  | 
| best_model.pkl |  | 
| metrics.json |  | 
| confusion_matrix.png |  | 
| shap_summary_beeswarm.png |  | 
| shap_summary_bar.png |  | 
| model_feature_importance.csv |  | 
| shap_top10.csv |  | 
| local_explanation_index_*.txt |  | 
| waterfall_index_*.png |  | 
| quick_report.json |  | 



📦 Requirements
- Python 3.8+
- Libraries:
- lightgbm
- scikit-learn
- shap
- pandas, numpy
- matplotlib
- joblib
Install dependencies:
pip install -r requirements.txt



🚀 Run the Pipeline
python main.py


Make sure your input data is preprocessed and available as X_train_proc, X_test_proc, y_train, y_test.

📊 SHAP Interpretation
- Global: Understand which features drive predictions across the dataset
- Local: Explain individual loan decisions with waterfall plots and plain-language summaries

📝 License
This project is for educational and research purposes. Commercial use requires permission.
