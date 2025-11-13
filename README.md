 # 📘 Credit Risk Classification and Model Optimization  

This project develops a complete machine learning pipeline to predict loan default risk using a structured credit dataset. The workflow covers data exploration, preprocessing, feature engineering, baseline modeling, hyperparameter tuning, ROC-based threshold optimization, and SHAP explainability. The final tuned models achieved a performance lift from **0.72 ROC-AUC (Logistic baseline)** to **0.89 (XGBoost/LightGBM tuned)**, offering both accuracy and interpretability for real-world credit risk applications.

---

## 🧹 Data Preprocessing  
- Capped unrealistic outliers (e.g., age > 100, employment length > 80) and applied IQR clipping for skew-heavy variables like income and loan amount.  
- Imputed missing values using **median** (numeric) and **mode** (categorical).  
- One-Hot Encoded categorical features to make the dataset fully numerical for tree-based models.  
- Addressed class imbalance through **class weighting**, improving recall for minority “default” cases.

---

## 🤖 Modeling Approach  

### **Logistic Regression – Baseline**  
Established the baseline model with ROC-AUC ~0.72.

### **Random Forest**  
Provided improved predictive performance with ensemble bagging; later tuned using Optuna.

### **XGBoost**  
Delivered strong gains due to gradient boosting; further improved through optimized depth, learning rate, and subsampling.

### **LightGBM**  
Achieved top performance with fast training and leaf-wise tree growth, reaching ROC-AUC ~0.89 after tuning.

---

## 🎯 Threshold Optimization  
Applied ROC-based threshold selection methods to better align predictions with business goals:  
- **Youden’s J statistic**  
- **Closest-to-(0,1)** ROC point  
- Optional recall-priority thresholds  

These improved the model’s ability to detect high-risk borrowers early.

---

## 🔍 SHAP Explainability  
SHAP values were used to interpret key drivers of default risk, including:  
- Income-to-loan ratio  
- Loan amount  
- Employment length  
- Credit bureau history  

These insights enhance transparency and model trustworthiness for financial decision-making.

---

## 🧠 Key Skills Demonstrated  
- End-to-end ML pipeline development  
- Preprocessing and feature engineering  
- Random Forest, XGBoost, LightGBM modeling  
- Optuna-based hyperparameter tuning  
- ROC-based thresholding  
- SHAP interpretability

