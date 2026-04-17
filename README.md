# Loan Default Prediction Project

## 1. **Business Understanding**

### **Problem Statement**
A financial institution wants to predict whether a new customer will default on a loan, based on their historical records and various personal features. The goal is to reduce the financial risks involved in lending by accurately forecasting whether a borrower will default or not.

### **Business Goal**
The goal is to develop and compare multiple machine learning models to predict the likelihood of loan default. These models will help the bank assess the risk of default, optimize lending strategies, and ensure financial stability.

### **Solution Overview**
We will implement and evaluate four different classification algorithms to predict loan default:
- K‑Nearest Neighbors (KNN)
- Decision Tree
- Logistic Regression

These models will be evaluated on various performance metrics including accuracy, precision, recall, F1‑score, and log loss.

---

## 2. **Data Understanding**

### **Dataset Description**
The dataset consists of information about 346 loan applicants, including:
- **Loan Features**: Principal, terms, effective and due dates
- **Customer Features**: Age, gender, education level
- **Target Variable**: `loan_status` (whether the loan was defaulted or not)

### **Target Variable**
- `loan_status`: A binary variable indicating whether the loan was defaulted (1) or paid off (0).

### **Feature Types**
| Feature Type | Example |
|--------------|---------|
| Numerical    | Age, Loan Amount, Loan Terms |
| Categorical  | Education, Gender |
| Date         | Effective Date, Due Date |

---

## 3. **Data Preparation**

### **Cleaning Steps**
1. Convert date columns to proper datetime objects.
2. Handle missing or inconsistent values.
3. Convert categorical variables into numerical form using one-hot encoding.
4. Ensure no missing values or inconsistent formatting in the dataset.

### **Feature Engineering**
1. Extract relevant features from the date columns (e.g., year, month).
2. Encode categorical variables using techniques like one-hot encoding.
3. Standardize numerical features for modeling.

---

## 4. **Data Preprocessing**

### **Feature Scaling**
The numerical features are standardized using **StandardScaler** to normalize the values so that they all contribute equally to the models:

X = preprocessing.StandardScaler().fit(X).transform(X)

### 4. Train‑Test Split
- The dataset is split into **80% training** and **20% testing** using `train_test_split`.

## 5. Modeling
The following models were implemented and trained:

### K‑Nearest Neighbors (KNN)
- Tuned hyperparameter **k** using cross‑validation.
- The optimal value of **k = 7**.

### Decision Tree
- Built a decision tree and tuned the **maximum depth** of the tree.
- Visualized the tree using **Graphviz**.

### Support Vector Machine (SVM)
- Evaluated multiple kernels including **linear** and **RBF**.

### Logistic Regression
- Optimized the **solver** and **penalty** parameters.
- Evaluated the model based on **predicted probabilities**.

## 6. Evaluation

### Metrics
Models were evaluated using the following metrics:
- **Jaccard Similarity**: Measures the similarity between predicted and actual outcomes.
- **F1‑Score**: The balance between precision and recall.
- **Log Loss**: Measures the accuracy of predicted probabilities.

### Comparison Table
| Algorithm            | Jaccard | F1‑Score | Log Loss |
|----------------------|---------|----------|----------|
| KNN                  | 0.757   | 0.726    | NA       |
| Decision Tree        | 0.742   | 0.705    | NA       |
| Logistic Regression  | 0.700   | 0.673    | 0.672    |

### Insights
- **KNN** provided the best Jaccard similarity scores.
- **Logistic Regression** performed the weakest, particularly in terms of Log Loss.

## 7. Insights & Conclusions

### Key Findings
- KNN performed well in terms of prediction accuracy and generalization.
- The Logistic Regression model, while useful, was less effective in predicting default probabilities.
- Decision Trees performed adequately but not as well as KNN and SVM.

## 8. Future Work

### Improvements
- Test ensemble models like **Random Forest** or **Gradient Boosting**.
- Perform additional **feature engineering** (e.g., interaction terms, feature selection).
- Address **class imbalance** if it exists in the dataset.
- Evaluate using **ROC/AUC curves** for model performance comparison.
