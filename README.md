# Wine Quality Prediction using Machine Learning

## Problem Statement

This project explores different ways to predict **wine quality** using physicochemical properties of red wine. The original dataset contains integer quality scores ranging from **3 to 8**. Each team member framed a unique machine learning problem based on this dataset:

- **Approach 1**: Frame the task as a **binary classification** — classifying wine as either "Good" or "Not Good"
- **Approach 2**: Frame the task as a **3-class classification** — "Bad", "Average", and "Good" wines
- **Approach 3 (Ayushmaan)**: Predict the **exact wine quality score (3–8)** as a multi-class classification problem

All three approaches aim to evaluate different ways of interpreting the same data and determining which model setup offers the best generalization and interpretability.

## Dataset

- Source: UCI Wine Quality Dataset
- File used: `winequality-red.csv`
- Target: `quality` (integer from 3 to 8)
- Features: 11 physicochemical test results per wine (e.g., alcohol, pH, sulphates)

## Methodology Overview

Across the three approaches, the general process followed was:

1. **Data Cleaning & Preprocessing**
   - Removed duplicates and cleaned column names
   - Encoded target variables according to task-specific labels
2. **Exploratory Data Analysis (EDA)**
   - Checked class distributions
   - Analyzed feature correlations and patterns with quality
3. **Modeling**
   - Tried various classifiers including XGBoost and Decision Trees
   - Evaluated performance on both balanced and imbalanced test sets
4. **Upsampling Techniques**
   - Compared **random oversampling** vs **SMOTE** for class balancing
5. **Evaluation**
   - Used accuracy, precision, recall, and F1-score
   - Confusion matrices and classification reports

---

## Results Summary

### Approach 1

### Approach 2

### Approach 3 – Multi-class Classification (Quality 3–8)
Explored wine quality prediction as a **multi-class classification problem**, where each class represents a specific quality score between 3 and 8.

#### Step 1: Linear Regression Baseline
Initially, I tested a Linear Regression model, treating wine quality as a continuous numerical value. However, this approach produced:
- Low R² and high RMSE/MAE
- Float predictions (e.g. 5.6), which were not ideal for the classification context
- Poor interpretability, especially when evaluating exact matches

This confirmed that regression was not suitable for the problem, and classification would be more effective.

#### Step 2: XGBoost Classifier Without Upsampling
I then trained an XGBoost classifier on the original (imbalanced) data. While it performed well on majority classes (like quality 5 and 6), it struggled to predict minority classes (like 3, 4, and 8), resulting in a bias toward frequent classes.

#### Step 3: Random Oversampling
To address the class imbalance, I applied random oversampling. This duplicated minority class samples to match the majority class size. The performance improved significantly, with:
- Accuracy rising to ~86.93%
- Balanced precision and recall for common classes
However, the model still showed instability for rare classes due to repeated data points and potential overfitting.

#### Step 4: SMOTE (Synthetic Minority Oversampling Technique)
Finally, I implemented SMOTE to generate synthetic data points for the minority classes. This led to:
- More diverse and meaningful training data
- Higher generalization and less overfitting
- Accuracy increasing to **94.31%** on the full original dataset
- F1-scores above 0.90 across nearly all classes

SMOTE proved to be the most effective upsampling method in this setup, and the final XGBoost model trained on SMOTE-augmented data delivered robust performance on both balanced and real-world test distributions.

| Method                | Accuracy (Original Data) | Macro F1 | Weighted F1 |
|-----------------------|--------------------------|----------|--------------|
| Random Oversampling   | 86.93%                   | 0.89     | 0.87         |
| SMOTE (Final Model)   | 94.31%                   | 0.93     | 0.94         |

SMOTE produced better generalization across rare and common classes, especially for qualities 3, 4, and 8.

## Key Insights

- **Alcohol**, **sulphates**, and **volatile acidity** are most correlated with higher wine quality
- Regression-based approaches underperform compared to classification methods
- SMOTE proved significantly more effective than random oversampling in handling class imbalance
- Evaluating on the full original dataset is more reflective of real-world performance than a small test split

---

## Contributions

### [Teammate Name – Binary Classification]


### [Teammate Name – Ternary Classification]


### Ayushmaan Kumar Yadav – Multi-class Classification (3 to 8)
- Framed and implemented the multi-class classification task for predicting wine quality scores (3 to 8)
- Applied SMOTE and random oversampling techniques for class balancing
- Trained and evaluated XGBoost classifiers
- Produced all evaluation reports, insights, and the final model comparison
