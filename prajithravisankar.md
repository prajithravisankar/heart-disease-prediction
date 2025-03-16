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