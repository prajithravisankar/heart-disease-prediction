### Sub-Todo 2.1.2: Feature Engineering
**Date:** March 16, 10:00 PM
**Performed By:** Emilio Santamaria

**Steps Taken**
**Installed Dependencies:**

- Installed pandas
- Installed scipy
- Installed scikit-learn
- Installed Data Wrangler

### Sub-Todo 2.2.1: Convert into X and Y 
Split Data into Features (X) and Target (Y):
- X Variables: Independent variables such as Age, Gender, Blood Pressure, and Heart Rate.
- Y Variable: Whether the patient has heart disease (Yes/No).

### Sub-todo 2.2.2: Scale and standardize numerical features

- Used StandardScaler to standardize numerical columns (mean = 0, variance = 1).
- Ignored boolean columns.
- Final Standardized columns: Age, Cholesterol, Blood Pressure, Heart Rate, Blood Sugar, Exercise Hours, Stress Level.
- Saved standardized values to 'standardized_data.csv'

### Sub-todo 2.2.3: Feature selection:

- Used SelectKBest to obtain the top 10 most relevant features for heart disease prediction.
- Used Recursive Feature Elimination (RFE) with logistic regression for feature selection.

### Insights:

- The top 4 most important features were the same for both methods.
- SelectKBest is more statistical, evaluating features independently, without considering interactions.
- RFE eliminates features iteratively, prioritizing those that perform well in combination.
- Given the nature of the Heart Disease case, RFE should be the preferred method for feature selection. To be discussed by group members how many features we want to consider for further analysis. 

### Saved Results:
Stored standardized data in 'standardized_data.csv' for reference and reproducibility.


### Sub-Todo 2.1.2: Feature Engineering & EDA
**Date:** March 22, 9:00 PM
**Performed By:** Emilio Santamaria

### New Sub-Todo : Fuzzy Columns (Feature Enginnering)
- Categorized columns like Cholesterol, Blood Pressure and Heart Rate into LOW, MEDIUM and HIGH levels 
- Added these new columns to the 'cleaned_data.csv'


### Sub-Todo ___: Multivariable Analysis 
- Elaborated the following graphs: 
- Heatmap comparing the correlation of all variables and presence of Heart Disease
- Heatmap comparing levels of Cholesterol, Blood Pressure, and Heart Rate, and presence of Heart Disease
- Box plot of relationship between Age, Levels of Cholesterol and presence of Heart Disease
- Heatmap comparing Cases of Heart Disease and combination of principal risks factors
- Barchart comparing Heart Disease cases based on different types of Anginas 


