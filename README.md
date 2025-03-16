# heart-disease-prediction
Predict heart disease risk using machine learning.

# PROJECT STRUCTURE
heart-disease-prediction/  
├── data/  
│   ├── raw/            # Contains project 2.csv ✔️  
│   └── processed/      # For cleaned data  
├── notebooks/          # For EDA and analysis  
│   └── data_cleaning.ipynb  
├── src/                # For preprocessing/modeling scripts  
│   └── data_preprocessing.py  
├── requirements.txt  
└── README.md  

### **Phase 1: Project Setup & Planning (March 13)**

**Goal**: Initialize repository, define roles, and finalize requirements.

- [ ]  **Main Todo 1.1: GitHub Setup**
    - [ ]  **Sub-todo 1.1.1**: Create a GitHub repository
    - [ ]  **Sub-todo 1.1.2**: Creat branches:
        - [ ]  `main`: Protected branch for final merges.
        - [x]  `dev`: Shared development branch for daily work.
        - [x]  `prajith-ravisankar`
        - [ ]  `emilio-santamaria` teammate has to create their branch and confirm.
- [ ]  **Main Todo 1.2: Requirements Finalization**
    - [ ]  **Sub-todo 1.2.1**: Review the PDF requirements and start working on:
        - Data cleaning steps (missing values, outliers).
        - Models to compare (e.g., Logistic Regression, Random Forest, XGBoost).
        - Metrics (accuracy, F1-score, AUC-ROC).
    - [x]  **Sub-todo 1.2.2**: confirm on communication platform (we are using discord)

### **Phase 2: Data Preparation (March 14–16)**

**Goal**: Clean and preprocess the dataset.

- [ ]  **Main Todo 2.1: Data Cleaning**
    - [ ]  **Sub-todo 2.1.1**: Load `project 2.csv` and inspect for:
        - Missing values (e.g., empty `Cholesterol` or `Blood Pressure` entries).
        - Duplicate rows.
        - what to do with outliers? not sure…
            - (e.g., `Age` > 100, `Blood Pressure` > 200).
        - etc…
    - [ ]  **Sub-todo 2.1.2**: Handle missing values:
        - 🆕 Use **KNNImputer** or **IterativeImputer** for advanced imputation (from the PDF).
        - Impute with mean/median or drop rows (document your choice).
        - etc…
    - [ ]  **Sub-todo 2.1.3**: Encode categorical variables:
        - `Gender`: Male=0, Female=1.
        - `Chest Pain Type`: Use **one-hot encoding** (avoids ordinal bias, per PDF).
        - etc…
- [ ]  **Main Todo 2.2: Feature Engineering**
    - [ ]  **Sub-todo 2.2.1**: Split data into features (`X`) and target (`y`).
    - [ ]  **Sub-todo 2.2.2**: Scale numerical features:
        - Use **StandardScaler** (Z-score normalization) for algorithms like SVM or Logistic Regression.
    - [ ]  **Sub-todo 2.2.3**: **Feature selection**:
        - Use **SelectKBest** or **RFE** (Recursive Feature Elimination) to reduce dimensionality.
    - [ ]  **Sub-todo 2.2.4**: Save preprocessed data as `cleaned_data.csv`.