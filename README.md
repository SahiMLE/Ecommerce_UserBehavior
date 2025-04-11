# Ecommerce_UserBehavior

This project explores user behavior in an e-commerce environment with the goal of predicting whether a session will lead to a purchase. Using the **Online Shoppers Intention Dataset**, several machine learning models were applied to classify user sessions based on clickstream and behavioral data.

---

## Objective

To build and evaluate classification models that predict purchasing intent from session data, enabling better insights for e-commerce optimization and targeting.

---

## Dataset

- **Source:** [Kaggle – Online Shoppers Intention Dataset](https://www.kaggle.com/datasets/syedhaideralizaidi/online-shoppers-intention)
- **Description:** Contains user session data across various e-commerce pages, including page values, bounce rates, exit rates, special days, traffic types, and visitor types.
- **Target Variable:** `Revenue` – whether a transaction was completed (True/False)

---

## Machine Learning Models Used

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Random Forest (optional)  
- Support Vector Machine (SVM)

---

## Workflow

1. **Data Cleaning**
   - Handling null values and type conversions
   - Encoding categorical variables

2. **Exploratory Data Analysis (EDA)**
   - Distribution of purchase vs. non-purchase sessions
   - Correlation heatmap and feature relationships

3. **Feature Engineering**
   - Scaling continuous features
   - Label encoding and one-hot encoding

4. **Model Building**
   - Train-test split
   - Classifier training
   - Performance comparison

5. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-Score
   - ROC-AUC
   - Confusion Matrix

---

## Tools & Technologies

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook

---

## Results

Logistic Regression and Decision Tree models offered strong predictive performance, with balanced precision and recall. Feature importance analysis highlighted key session metrics like bounce rate, exit rate, and page values in determining user intent.

---

## Future Enhancements

- Perform hyperparameter tuning using GridSearchCV  
- Try ensemble models (Gradient Boosting, XGBoost)  
- Address class imbalance using SMOTE or similar techniques  
- Deploy a real-time demo with Streamlit

---

## Author

**Sai Sahi**  
MSc Computer Science – Teesside University  
Email: sahisai141@gmail.com  
GitHub: [github.com/SahiMLE](https://github.com/SahiMLE)

---

*This project is part of my AI/ML portfolio. Contributions and feedback are welcome!*
