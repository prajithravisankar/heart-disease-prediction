Phase 3: Exploratory Data Analysis (EDA)  
Date: March 20, 11：00PM Performed By: Fangai Cheng

Main Todo 3.1: Univariate Analysis  
Sub-todo 3.1.1: Plot Distributions for:
- **Age, Cholesterol, Blood Pressure** (Histograms)
- **Heart Disease** (Pie chart for class balance)

Sub-todo 3.1.2: Document observations
- Example: *"30% of patients have heart disease."*

Main Todo 3.2: Bivariate/Multivariate Analysis  
Sub-todo 3.2.1: Correlation Heatmap  
- Visualizing correlation between features and **Heart Disease**  

Sub-todo 3.2.2: Boxplots  
Cholesterol vs. Heart Disease**

Sub-todo 3.2.3: Pairplot  
- **Key features** (e.g., Age, Blood Pressure)
Result: 

Phase 4: Model Development (March 18–20)
Goal: Train and compare baseline models.

Main Todo 4.1: Model Training
Sub-todo 4.1.1: Train 3 models:
- Logistic Regression.
- Random Forest.

Output for Logistic Regression:
Classification Report:
               precision    recall  f1-score   support

           0       0.87      0.92      0.89       122
           1       0.86      0.78      0.82        78
    accuracy                           0.86       200
   macro avg       0.86      0.85      0.86       200
weighted avg       0.86      0.86      0.86       200

Confusion Matrix:
 [[112  10] \n
 [ 17  61]]
AUC-ROC: 0.9508196721311475


Output for Random Forest:
Classification Report:
               precision    recall  f1-score   support

           0       0.99      1.00      1.00       122
           1       1.00      0.99      0.99        78
    accuracy                           0.99       200
   macro avg       1.00      0.99      0.99       200
weighted avg       1.00      0.99      0.99       200

Confusion Matrix:
 [[122   0]
 [  1  77]]
AUC-ROC: 1.0
