**Wine Type and Quality Classification - Machine Learning Project**

📌 Project Overview

The objective of this project was to identify and implement a machine learning classification algorithm that achieves the highest possible accuracy on a dataset from the UCI Machine Learning Repository.

Our workflow combined exploratory data analysis (EDA), data preprocessing, model benchmarking, and hyperparameter tuning to evaluate multiple classification algorithms and select the best-performing one.

🔎 Dataset

Source: UCI Machine Learning Repository

Criteria: Dataset selected with >1000 instances and ≥10 features.

Format: CSV file (included in project repo).

Preprocessing:

Handled missing values.

Standardized/normalized numerical features.

Encoded categorical variables.

Split into train/test sets for evaluation.

🛠️ Methodology
Models Tested

We evaluated a range of baseline classifiers to determine the best fit for the dataset:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

K-Nearest Neighbors (KNN)

XGBoost Classifier

Support Vector Machines (SVM)

Multi-Layer Perceptron (MLP Neural Network)

Evaluation Metrics

Primary Metric: Classification accuracy.

Additional Metrics: Precision, Recall, F1-Score, Confusion Matrix for deeper insights.

Validation: Stratified train-test split and k-fold cross-validation to reduce variance.

Accuracy Optimization

Feature Engineering: Scaling, encoding, and feature importance ranking.

Hyperparameter Tuning: GridSearchCV/RandomizedSearchCV for Random Forest, XGBoost, and SVM.

Ensemble Approaches: Tested bagging/boosting to improve generalization.

📊 Key Findings

Baseline Results: Random Forest and XGBoost consistently outperformed simpler models.

Final Best Model: XGBoost Classifier, achieving the highest accuracy after tuning.

Insights:

Feature scaling improved SVM and KNN performance.

Ensemble methods offered significant gains over single classifiers.

Neural Networks required more tuning but did not outperform XGBoost in this case.

🗂️ Deliverables

MBAN2_Group4_A2_ipynb5_.ipynb – Python notebook with all code (runs in Colab).

dataset.csv – Cleaned dataset used in the analysis.

video.mp4 – 8-minute presentation explaining methodology and results.

README.md – Project overview (this file).

🚀 Key Takeaways

Ensemble methods (Random Forest, XGBoost) achieved the best accuracy for this dataset.

Systematic preprocessing and hyperparameter tuning were critical to model performance.

The project demonstrates how to compare, evaluate, and optimize classification algorithms for real-world datasets.
