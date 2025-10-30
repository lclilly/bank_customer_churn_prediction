Bank Customer Churn Prediction

Goal: Predict which customers are likely to close their accounts so the bank can proactively intervene with retention strategies.

📁 Project Structure
├── Bank Customer Churn Prediction.csv    # Original dataset
├── train_churn_project.py                # Main training script
├── requirements.txt                      # Python dependencies
├── best_churn_model.pkl                  # Trained model pipeline (generated)
├── high_risk_watchlist_top100.csv        # Top 100 at-risk customers (generated)
├── model_summary.txt                     # Model performance metrics (generated)
├── Roc_Curve_Random_Forest.png           # ROC curve visualization (generated)
└── Feature_Importances_Random_Forest.png # Feature importance plot (generated)

🎯 Business Impact

Actionable Insights: Identifies the top 100 customers most likely to churn
Cost Savings: Enables targeted retention efforts, potentially saving $200K+ in customer lifetime value
Key Drivers: Age (31%), number of products (21%), and account balance (11%) are the strongest churn predictors

🔍 Approach
1. Exploratory Data Analysis

Analyzed feature distributions and correlations with churn
Identified class imbalance and key patterns

2. Data Preprocessing

Scaled numerical features (StandardScaler)
One-hot encoded categorical variables
Removed non-predictive identifiers

3. Modeling

Baseline: Logistic Regression
Final Model: Random Forest with GridSearchCV hyperparameter tuning
Optimized for ROC AUC to balance precision and recall

4. Evaluation

Confusion matrix analysis
ROC curve and AUC score
Feature importance ranking
Classification report (precision, recall, F1-score)

5. Deployment-Ready Outputs

Serialized model pipeline for production use
High-risk customer watchlist (CSV) for retention teams
Visual reports (ROC curve, feature importances)

📊 Model Performance
Model                ROC AUC
Logistic Regression  0.78
Random Forest        0.86

Top 3 Predictive Features:
Age (30.9%)
Number of Products (21.0%)
Account Balance (10.8%)

🚀 How to Run
Clone the repository:

bashgit clone https://github.com/lclilly/bank_customer_churn_prediction.git
cd bank_customer_churn_prediction

Install dependencies:

bashpip install -r requirements.txt

Run the training script:

bashpython train_churn_project.py
```

4. **Outputs generated:**
- `best_churn_model.pkl` - Trained model ready for deployment
- `high_risk_watchlist_top100.csv` - Priority customer list
- `model_summary.txt` - Performance metrics
- Visualization plots (ROC curve, feature importances)

## 📦 Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

💡 Future Improvements
1. Implement SMOTE or other techniques to handle class imbalance
2. Experiment with ensemble methods (XGBoost, LightGBM)
3. Adjust classification threshold for cost-sensitive learning
4. Deploy model as a REST API or integrate with business dashboard
5. Add model monitoring and retraining pipeline

👤 Author
Lilian Cheuno

Note: This project demonstrates end-to-end machine learning workflow from data preprocessing to deployment-ready outputs, with a focus on actionable business insights.
