# Wine Quality Prediction using Machine Learning

## Problem Statement

This project explores different ways to predict **wine quality** using physicochemical properties of red wine. The original dataset contains integer quality scores ranging from **3 to 8**. Each team member framed a unique machine learning problem based on this dataset:

- **Approach 1 (Diego)**: Frame the task as a **binary classification** — classifying wine as either "Good" or "Not Good"
- **Approach 2 (Huajia)**: Frame the task as a **3-class classification** — "Bad", "Average", and "Good" wines
- **Approach 3 (Ayushmaan)**: Predict the **exact wine quality score (3–8)** as a multi-class classification problem

All three approaches aim to evaluate different ways of interpreting the same data and determining which model setup offers the best generalization and interpretability.

## Dataset

- Source: UCI Wine Quality Dataset
- File used: `winequality-red.csv` (https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data)
- Target: `quality`
- Features: 11 physicochemical test results per wine (e.g., alcohol, pH, sulphates)

## Methodology Overview

Across the three approaches, the general process followed was:

1. **Data Cleaning & Preprocessing**
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

### Approach 2 - Multi-class Classification ("Not Good", "Average", and "Good")
Explored red wine quality prediction as a multi-class classification problem, where each wine sample was categorized into three quality levels: 
- Not good
- Average
- Good

#### Step 1: Exploratory Data Analysis (EDA)
I first started by exploring the red wine dataset as a multi-class classification problem, where I grouped wine samples into three categories: **not good, average, and good**, based on their quality scores. I examined the distribution of each feature and the class labels, and discovered a clear class imbalance, with most samples labeled as “average.” Through visualizations like histograms, boxplots, and correlation matrices, I identified **alcohol, volatile acidity**, and **sulphates** as the most important features affecting wine quality.

#### Step 2: Baseline Classification with Imbalanced Data
Then, I built a baseline using Trained Random Forest and XGBoost classifiers on the original (imbalanced) dataset:
- Achieved decent accuracy **(~83%)** but performance was skewed toward the majority class ("average")
- Minority classes ("not good" and "good") had **lower recall** and **F1-scores**
- Indicated a need to address class imbalance for fairer predictions

#### Step 3: SMOTE (Synthetic Minority Oversampling Technique)
To tackle this, I applied **SMOTE** (Synthetic Minority Oversampling Technique) to generate synthetic samples for the minority classes in the training set. Unlike random oversampling, SMOTE creates more diverse data points, reducing overfitting and improving generalization.

#### Step 4: Final Model Evaluation
After retraining the models on the SMOTE-balanced data, I observed a significant improvement in performance. The final XGBoost model achieved an accuracy of **86.6%** on the test set, with more balanced precision, recall, and F1-scores across all three quality categories. The confusion matrix confirmed that the model was no longer biased toward the majority class and handled all classes more fairly.

| Method                  | Accuracy (Original Data) | Macro F1 | Weighted F1 |
|-------------------------|--------------------------|----------|--------------|
| Imbalanced Data (Baseline) | ~83.00%                  | Lower for minority classes | Moderate     |
| SMOTE (Final Model)        | **86.60%**                | Improved                  | Balanced     |

## Key Insights

- Using three quality categories ("not good", "average", and "good") provided a more nuanced and realistic view of wine quality distribution compared to a simple binary classification (e.g., "good" vs. "not good").
- A binary approach might simplify the modeling process and boost headline accuracy, but it would also result in **loss of valuable granularity**, especially in distinguishing borderline wines (typically the “average” class).
- Training and testing on three categories helped highlight **class imbalance issues**, which were critical to solve using techniques like **SMOTE**. These problems would be hidden in a binary setup.
- The model’s ability to separate average wines from both good and poor ones reflects greater precision, especially useful for wine producers or quality control systems aiming for fine-grained decisions rather than just pass/fail.
- In real-world applications (e.g., quality grading, pricing, recommendation systems), **three-tier classification offers more actionable insights** than a binary prediction.

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


### Yu Huajia (33.3%) - Multi-class Classification ("Not Good", "Average", and "Good")
- Framed and implemented a multi-class classification task for predicting red wine quality (categorized as not good, average, good)
- Performed EDA to identify key features and class imbalance issues
- Applied SMOTE to balance minority classes and improve model fairness
- Trained and evaluated Random Forest and XGBoost classifiers
- Compared model performance before and after class balancing
- Generated evaluation metrics, visualizations, and final model insights

### Ayushmaan Kumar Yadav (33.3%) – Multi-class Classification (3 to 8)
- Framed and implemented the multi-class classification task for predicting wine quality scores (3 to 8)
- Applied SMOTE and random oversampling techniques for class balancing
- Trained and evaluated XGBoost classifiers
- Produced all evaluation reports, insights, and the final model comparison
