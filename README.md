# heart-disease-prediction
Predict heart disease risk using machine learning.

Here's a clean **project structure** for your `README.md` (formatted in Markdown):

```markdown
# Project Structure
heart-disease-prediction/  
├── **data/**  
│   ├── **raw/**               # Raw dataset (`project 2.csv`)
│   └── **processed/**         # Cleaned/preprocessed data (e.g., `cleaned_data.csv`)  
├── **notebooks/**             # Jupyter notebooks for analysis  
│   ├── `eda.ipynb`            # Exploratory Data Analysis  
│   └── `model_training.ipynb` # Model experiments and evaluation  
├── **src/**                   # Python scripts  
│   ├── `data_preprocessing.py` # Data cleaning/feature engineering  
│   ├── `train_model.py`        # Model training and tuning  
│   └── `app.py`                # (Optional) CLI/Flask deployment if we have time
├── **models/**                # Trained models (e.g., `random_forest.pkl`)  
├── **results/**               # Visualizations, metrics, and reports  
│   ├── `confusion_matrix.png`  
│   └── `feature_importance.png`  
├── **requirements.txt**      # Python dependencies  
├── **README.md**             # Project overview and instructions  
└── **OTHERS**               # MIT License (or others if we need any)  
```

### **Key Directories Explained**  
- **`data/raw`**: Contains the original unprocessed dataset.  
- **`data/processed`**: Stores cleaned data after preprocessing.  
- **`notebooks`**: For exploratory analysis and model prototyping.  
- **`src`**: Reusable Python scripts for data cleaning, modeling, and deployment.  
- **`models`**: Saves trained models for later use.  
- **`results`**: Stores visualizations, performance metrics, and reports.  

### **Collaboration Notes**  
- Use branches `prajith-ravisankar` and `emilio-santamaria` and `lasombra7` 
- Merge changes into `dev` for daily collaboration after we agree on.  
- Final stable code goes into `main` (protected branch).

### **Phase 1: Project Setup & Planning (March 13)**

**Goal**: Initialize repository, define roles, and finalize requirements.

- [x]  **Main Todo 1.1: GitHub Setup**
    - [x]  ~~**Sub-todo 1.1.1**: Create a GitHub repository~~
    - [x]  ~~**Sub-todo 1.1.2**: Creat branches:~~
        - [x]  ~~`main`: Protected branch for final merges.~~
        - [x]  ~~`dev`: Shared development branch for daily work.~~
        - [x]  ~~`prajith-ravisankar`~~
        - [ ]  `emilio-santamaria` teammate has to create their branch and confirm.
        - [x] `lasombra7` teammate has to accept the invite from Github and create their branch to start contributions
- [x]  ~~**Main Todo 1.2: Requirements Finalization**~~
    - [x]  ~~**Sub-todo 1.2.1**: Review the PDF requirements and start working on:~~
        ~~- Data cleaning steps (missing values, outliers).~~
        ~~- Models to compare (e.g., Logistic Regression, Random Forest, XGBoost).~~
        ~~- Metrics (accuracy, F1-score, AUC-ROC).~~
    - [x]  ~~**Sub-todo 1.2.2**: confirm on communication platform (we are using discord)~~

### **Phase 2: Data Preparation (March 14–16)**

**Goal**: Clean and preprocess the dataset.

- [x]  ~~**Main Todo 2.1: Data Cleaning**~~
    - [x]  ~~**Sub-todo 2.1.1**: Load `project 2.csv` and inspect for:~~
        - ~~Missing values (e.g., empty `Cholesterol` or `Blood Pressure` entries).~~
        - ~~Duplicate rows.~~
        - ~~what to do with outliers? not sure…~~
            - ~~(e.g., `Age` > 100, `Blood Pressure` > 200).~~
        - ~~etc~~…
    - [x]  ~~**Sub-todo 2.1.2**: Handle missing values:~~
        - ~~Use **KNNImputer** or **IterativeImputer** for advanced imputation~~
    - [x]  ~~**Sub-todo 2.1.3**: Encode categorical variables:~~
        - ~~`Gender`: Male=0, Female=1.~~
- [ ]  **Main Todo 2.2: Feature Engineering**
    - [ ]  **Sub-todo 2.2.1**: Split data into features (`X`) and target (`y`).
    - [ ]  **Sub-todo 2.2.2**: Scale numerical features:
        - Use **StandardScaler** (Z-score normalization) for algorithms like SVM or Logistic Regression.
    - [ ]  **Sub-todo 2.2.3**: **Feature selection**:
        - Use **SelectKBest** or **RFE** (Recursive Feature Elimination) to reduce dimensionality.
    - [ ]  **Sub-todo 2.2.4**: Save preprocessed data as `cleaned_data.csv`.

---

### **Phase 3: Exploratory Data Analysis (March 16–17)**

**Goal**: Generate insights and visualizations.

- [ ]  **Main Todo 3.1: Univariate Analysis**
    - [ ]  **Sub-todo 3.1.1**: Plot distributions for:
        - `Age`, `Cholesterol`, `Blood Pressure` (histograms).
        - `Heart Disease` (pie chart for class balance).
    - [ ]  **Sub-todo 3.1.2**: Document observations (e.g., "30% of patients have heart disease").
- [ ]  **Main Todo 3.2: Bivariate/Multivariate Analysis**
    - [ ]  **Sub-todo 3.2.1**: Correlation heatmap (features vs. `Heart Disease`).
    - [ ]  **Sub-todo 3.2.2**: Boxplots for `Cholesterol` vs. `Heart Disease`.
    - [ ]  **Sub-todo 3.2.3**: Pairplot for key features (e.g., `Age`, `Blood Pressure`).

---

### **Phase 4: Model Development (March 18–20)**

**Goal**: Train and compare baseline models.

- ~~[x]  **Main Todo 4.1: Model Training**~~
    - [x]  ~~**Sub-todo 4.1.1**: Train 3 models:~~
        - ~~Logistic Regression.~~
        - ~~Random Forest.~~
        - ~~XGBoost.~~
    - [x]  ~~**Sub-todo 4.1.2**: Use `train_test_split` (80-20 split).~~
- [x]  ~~**Main Todo 4.2: Baseline Evaluation**~~
    - [x]  ~~**Sub-todo 4.2.1**: Calculate metrics:~~
        - ~~Accuracy, Precision, Recall, F1-score, AUC-ROC.~~
    - [x]  ~~**Sub-todo 4.2.2**: Document results in a shared spreadsheet.~~

---

### **Phase 5: Model Optimization (March 21–22)**

**Goal**: Improve model performance.

- [ ]  **Main Todo 5.1: Hyperparameter Tuning**
    - [ ]  **Sub-todo 5.1.1**: Use `GridSearchCV` or `RandomizedSearchCV` for:
        - Random Forest (tune `n_estimators`, `max_depth`).
        - XGBoost (tune `learning_rate`, `max_depth`).
    - [ ]  **Sub-todo 5.1.2**: Re-evaluate metrics post-tuning.
- [ ]  **Main Todo 5.2: Feature Importance**
    - [ ]  **Sub-todo 5.2.1**: Plot feature importance for the best model.
    - [ ]  **Sub-todo 5.2.2**: Identify top 5 risk factors (e.g., `Cholesterol`, `Age`).

---

### **Phase 6: Deployment & Documentation (March 23)**

**Goal**: Prepare a simple deployment and final report.

- [ ]  **Main Todo 6.1: Deployment**
    - [ ]  **Sub-todo 6.1.1**: Create a `predict()` function for new data.
    - [ ]  **Sub-todo 6.1.2**: Build a basic CLI or Flask app for predictions.
