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