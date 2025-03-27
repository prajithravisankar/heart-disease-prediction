### Sub-Todo 2.1.1: Data Inspection (Outliers, Missing Values, Duplicates)
**Date**: March 16  6:00 PM
**Performed By**: Prajith Ravisankar  

#### **Steps Taken**:  
1. **Loaded Dataset**:  
   - Path: `data/raw/project 2.csv`  
   - Verified columns: `Age`, `Cholesterol`, `Blood Pressure`, etc.  

2. **Checked for Missing Values**:  
   - Used `df.isnull().sum()` to count missing values in each column.  
   - Found significant missing values in `Alcohol Intake` (340 entries).  

3. **Checked for Duplicates**:  
   - Used `df.duplicated().sum()` to count duplicate rows.  
   - Found **0 duplicate rows**.  

4. **Detected Outliers**:  
   - Used Z-score method (threshold ±3) for numerical columns: `Age`, `Cholesterol`, `Blood Pressure`, `Heart Rate`, `Blood Sugar`, `Stress Level`.  
   - Found **0 outliers** beyond the ±3 threshold.  

#### **Results**:  
- **Missing Values**:  
  - `Alcohol Intake`: 340 missing entries (needs imputation).  
  - Other columns: No missing values.  
- **Duplicates**: 0 duplicates found.  
- **Outliers**: No extreme outliers detected in numerical features.  

#### **Next Steps**:  
- Proceed to handle missing values in `Alcohol Intake` (Sub-Todo 2.1.2).  
- Use advanced imputation methods like KNNImputer or IterativeImputer (per PDF requirements).  

---

### Sub-Todo 2.1.2: Handle Missing Values
**Date**: March 16  6:35 PM
**Performed By**: Prajith Ravisankar  

#### Steps Taken:
1. Used **KNNImputer** to fill missing values in:
   - `Cholesterol` (1 missing value).
   - `Blood Pressure` (4 missing values).
   - `Alcohol Intake` (340 missing values).
2. Temporarily encoded categorical variables for KNNImputer.
3. Saved the cleaned dataset to `../data/processed/cleaned_data.csv`.

#### Why KNNImputer?
- It uses relationships between features to estimate missing values.
- Better than mean/median imputation for datasets with patterns.

---

### Sub-Todo 2.1.3: Encode Categorical Variables
**Date**: March 16  6:50 PM
**Performed By**: Prajith Ravisankar  

#### Steps Taken:
1. **Checked `Gender` Column**:
   - Found that `Gender` was already one-hot encoded into a binary column named `Gender_Male`.
   - `Gender_Male=True` represents Male, and `Gender_Male=False` represents Female.
   - No further encoding needed for `Gender`.

2. **One-Hot Encoded `Chest Pain Type`**:
   - Used `pd.get_dummies()` to create binary columns for each category.
   - Dropped the first category to avoid multicollinearity.

3. **Handled Other Categorical Variables**:
   - Verified that other categorical variables (e.g., `Smoking`, `Alcohol Intake`) were already one-hot encoded.
   - No additional encoding required.

### Sub-Todo 2.2.1 (continuation): Split Features and Target  (Please use **cleaned_data.csv** not encoded_data.csv)
**Date**: March 16  7:30 PM
**Performed By**: Prajith Ravisankar  

**Steps Taken**:  
1. Loaded `cleaned_data.csv` (post-imputation and encoding).  
2. Separated features (`X`) and target (`y`).  
3. Verified shapes:  
   - `X` shape: (n_samples, n_features)  
   - `y` shape: (n_samples,)  

**Why This Matters**:  
- Models require features and labels to be split for training/testing.  
- Ensures the target variable (`Heart Disease`) is isolated from input data.
- Encoding ensures that machine learning models can process categorical data.
- One-hot encoding avoids ordinal bias (per PDF requirements).
- Please use **cleaned_data.csv** not encoded_data.csv

---

# Phase 3: Exploratory Data Analysis (March 16–22, 2025)

## Sub-Todo 3.1.1: Expanded Univariate Analysis
**Date**: March 16–21, 2025  
**Performed By**: Prajith Ravisankar  

### Key Analyses Performed:
1. **Numerical Feature Distributions**:
   - Created histograms for all 7 numerical features:
     - Age, Cholesterol, Blood Pressure  
     - Heart Rate, Exercise Hours  
     - Stress Level, Blood Sugar  
   - Used Z-score normalized values from `standardized_data.csv`
   - Key Insight: All features show near-normal distribution post-standardization

2. **Categorical/Dummy Variable Analysis**:
   - Generated bar plots for 11 categorical features:
     ```python
     ["Gender_Male", "Smoking_Former", "Smoking_Never", 
      "Alcohol Intake_Moderate", "Family History_Yes",
      "Diabetes_Yes", "Obesity_Yes", "Exercise Induced Angina_Yes",
      "Chest Pain Type_Atypical Angina", 
      "Chest Pain Type_Non-anginal Pain",
      "Chest Pain Type_Typical Angina"]
     ```
   - Added percentage labels for easier interpretation

3. **Enhanced Target Analysis**:
   - Combined visualization showing:
     - Class distribution pie chart (52.3% Heart Disease)
     - Blood Pressure vs Heart Disease boxplot
   - Key Finding: Higher BP correlates with heart disease presence

4. **Additional Insights**:
   - Stress Level Distribution by Gender (violin plot)
   - Cholesterol vs Diabetes Status (boxen plot)

#### Generated Visualizations is in results folder
``` markdown
results/categorical_features_distributions.png
results/cholesterol_diabetes.png
results/class_balance.png
results/feature_distributions.png
results/numerical_features_distributions.png
results/stress_by_gender.png
results/target_analysis.png
```
---

### Phase 4: Data Cleaning & Model Implementation Report  (XGboost)
**Date**: March 26 10:00 PM  
**Performed By**: Prajith Ravisankar  

---

### **File 1: data_cleaning.py**  
**Purpose**: Raw data preprocessing pipeline  

#### **Key Functions**:  
1. `load_and_clean_data()`:  
   - **Input**: `project 2.csv` (raw data)  
   - **Operations**:  
     - Removed 0 duplicates  
     - Imputed 340 missing `Alcohol Intake` values with KNNImputer (k=5 neighbors)  
     - Encoded 8 categorical features using LabelEncoder  
     - Clipped outliers via IQR method (1.5x bounds) for all numeric columns  
   - **Output**: `cleaned_data.csv`  

2. `save_cleaned_data()`:  
   - Saves processed data to `/data/processed`  

#### **Problems Faced**:  
- High missing values (340) in `Alcohol Intake` - Risk of biased imputation  
- All categorical features encoded as ordinal (may not reflect true relationships)  
- No outliers detected - Possible over-clipping of natural variation  

---

### **File 2: feature_engineering.py**  
**Purpose**: Feature transformation & selection  

#### **Key Functions**:  
1. `engineer_features()`:  
   - StandardScaler applied to 15 features  
   - Selected top 8 features via ANOVA F-test (`SelectKBest`)  
   - Final Features: Age, Gender, Cholesterol, Alcohol Intake, Family History, Diabetes, Obesity, Chest Pain Type  

2. `split_data()`:  
   - 80/20 stratified split (800/200 samples)  
   - Preserved class distribution in splits  

#### **Problems Faced**:  
- Feature selection only used univariate methods (missed multivariate interactions)  
- No creation of derived features (e.g., BMI, BP ratios)  
- Target leakage risk with full-dataset scaling  

---

### **File 3: xboost_training.py**  
**Purpose**: Model training & evaluation  

#### **Key Components**:  
1. Hyperparameter Tuning:  
   - GridSearchCV with 243 combinations  
   - Best Params: `{'colsample_bytree': 0.6, 'learning_rate': 0.3, 'max_depth': 3}`  

2. Evaluation Metrics:  
   ```python
   Accuracy: 0.5650  # Barely better than random
   Precision: 0.4328  # High false positives
   Recall: 0.3718     # Misses 63% of true cases
   AUC-ROC: 0.5269    # Minimal discrimination power