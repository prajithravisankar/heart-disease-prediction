import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_cleaned_data(input_path):
    """
    Load the cleaned dataset

    Parameters:
    input_path (str): Path to the cleaned CSV file

    Returns:
    pd.DataFrame: Cleaned dataset
    """
    print(f"Loading cleaned data from {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Verify target variable
    print(f"\nTarget variable distribution:")
    print(df['Heart Disease'].value_counts())
    print(f"Class balance: {df['Heart Disease'].mean() * 100:.2f}% positive cases")

    return df


def create_new_features(df):
    """
    Create new engineered features from existing ones

    Parameters:
    df (pd.DataFrame): Input dataframe with cleaned data

    Returns:
    pd.DataFrame: Dataframe with additional engineered features
    """
    print("\nCreating new features...")
    df_new = df.copy()

    # 1. Age Categories (risk increases with age)
    age_bins = [0, 40, 50, 60, 80]
    age_labels = ['Young Adult', 'Middle Aged', 'Senior', 'Elderly']
    df_new['Age_Group'] = pd.cut(df_new['Age'], bins=age_bins, labels=age_labels)

    # Convert to one-hot encoding
    age_dummies = pd.get_dummies(df_new['Age_Group'], prefix='Age_Group')
    df_new = pd.concat([df_new, age_dummies], axis=1)
    df_new.drop('Age_Group', axis=1, inplace=True)

    # 2. BP Category (based on clinical guidelines)
    df_new['BP_Category'] = pd.cut(
        df_new['Blood Pressure'],
        bins=[0, 120, 140, 160, 200],
        labels=['Normal', 'Prehypertension', 'Stage1', 'Stage2']
    )
    bp_dummies = pd.get_dummies(df_new['BP_Category'], prefix='BP')
    df_new = pd.concat([df_new, bp_dummies], axis=1)
    df_new.drop('BP_Category', axis=1, inplace=True)

    # 3. Cholesterol Risk Category
    df_new['Chol_Category'] = pd.cut(
        df_new['Cholesterol'],
        bins=[0, 200, 240, 400],
        labels=['Desirable', 'Borderline', 'High']
    )
    chol_dummies = pd.get_dummies(df_new['Chol_Category'], prefix='Chol')
    df_new = pd.concat([df_new, chol_dummies], axis=1)
    df_new.drop('Chol_Category', axis=1, inplace=True)

    # 4. Combined Risk Factors (count of key risk factors)
    # Check if the columns exist in one-hot encoded format
    risk_cols = []
    for prefix in ['Obesity_Y', 'Smoking_', 'Diabetes_Y', 'Family History_Y']:
        matching_cols = [col for col in df_new.columns if col.startswith(prefix)]
        risk_cols.extend(matching_cols)

    # If we have found some risk columns in our expected format
    if risk_cols:
        df_new['Risk_Factor_Count'] = df_new[risk_cols].sum(axis=1)

    # 5. BMI Proxy (experimental - if we have height and weight)
    # Note: Since we don't have direct BMI components, we'll skip this

    # 6. Interaction Features
    # Interaction between Age and Blood Pressure
    df_new['Age_BP_Interaction'] = df_new['Age'] * df_new['Blood Pressure'] / 100

    # Interaction between Cholesterol and Blood Pressure
    df_new['Chol_BP_Interaction'] = df_new['Cholesterol'] * df_new['Blood Pressure'] / 1000

    print(f"Created {df_new.shape[1] - df.shape[1]} new features")
    print(f"New dataset shape: {df_new.shape}")

    return df_new


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets (BEFORE preprocessing)

    Parameters:
    df (pd.DataFrame): Input dataframe
    test_size (float): Proportion of test set
    random_state (int): Random seed for reproducibility

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    print("\nSplitting data into training and testing sets...")

    # Separate features and target
    X = df.drop('Heart Disease', axis=1)
    y = df['Heart Disease']

    # Split into train/test sets (stratify ensures class balance is maintained)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples ({y_train.mean() * 100:.2f}% positive)")
    print(f"Testing set: {X_test.shape[0]} samples ({y_test.mean() * 100:.2f}% positive)")

    return X_train, X_test, y_train, y_test


def preprocess_features(X_train, X_test):
    """
    Scale numerical features (fit on train, transform both)

    Parameters:
    X_train (pd.DataFrame): Training features
    X_test (pd.DataFrame): Testing features

    Returns:
    tuple: Scaled X_train, X_test, list of numerical columns, scaler object
    """
    print("\nPreprocessing features...")

    # Identify numerical columns (exclude any one-hot encoded or binary columns)
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Display numerical columns being scaled
    print(f"Scaling {len(numerical_cols)} numerical columns:")
    print(", ".join(numerical_cols))

    # Initialize scaler
    scaler = StandardScaler()

    # Create copies to avoid modifying originals
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    # Fit scaler on training data, transform both train and test
    X_train_processed[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_processed[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train_processed, X_test_processed, numerical_cols, scaler


def select_features(X_train, y_train, X_test, n_features=10):
    """
    Select best features using multiple methods and ensemble the results

    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training target
    X_test (pd.DataFrame): Testing features
    n_features (int): Number of features to select

    Returns:
    tuple: X_train_selected, X_test_selected, selected_features
    """
    print(f"\nSelecting top {n_features} features using multiple methods...")

    # Method 1: ANOVA F-value (good for linear relationships)
    selector_f = SelectKBest(score_func=f_classif, k=n_features)
    selector_f.fit(X_train, y_train)
    f_support = selector_f.get_support()

    # Method 2: Mutual Information (captures non-linear relationships)
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features)
    selector_mi.fit(X_train, y_train)
    mi_support = selector_mi.get_support()

    # Method 3: Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_
    rf_indices = np.argsort(rf_importances)[::-1][:n_features]
    rf_support = np.zeros(X_train.shape[1], dtype=bool)
    rf_support[rf_indices] = True

    # Method 4: Recursive Feature Elimination with Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rfe = RFE(estimator=lr, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    rfe_support = rfe.support_

    # Create feature selection votes dataframe
    feature_names = X_train.columns.tolist()
    votes = pd.DataFrame({
        'Feature': feature_names,
        'F-Score': f_support,
        'Mutual Info': mi_support,
        'Random Forest': rf_support,
        'RFE': rfe_support
    })

    # Sum the votes to find consensus features
    votes['Total Votes'] = votes.iloc[:, 1:].sum(axis=1)
    votes = votes.sort_values('Total Votes', ascending=False)

    # Select features with at least 2 votes
    consensus_features = votes[votes['Total Votes'] >= 2]['Feature'].tolist()

    # If we got too few features, take the top n by total votes
    if len(consensus_features) < n_features:
        consensus_features = votes.nlargest(n_features, 'Total Votes')['Feature'].tolist()
    # If we got too many, take the top n
    elif len(consensus_features) > n_features:
        consensus_features = consensus_features[:n_features]

    print("\nSelected features (by consensus of 4 methods):")
    for i, feature in enumerate(consensus_features, 1):
        print(f"{i}. {feature}")

    # Create plots directory
    plots_dir = "../results"
    os.makedirs(plots_dir, exist_ok=True)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    votes_for_plot = votes.nlargest(min(20, len(votes)), 'Total Votes')
    ax = sns.barplot(
        x='Total Votes',
        y='Feature',
        data=votes_for_plot,
        palette='viridis'
    )
    plt.title('Feature Selection Consensus (Votes from 4 Methods)', fontsize=14)
    plt.xlabel('Votes (max 4)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/feature_selection_votes.png", dpi=300)
    plt.close()

    # Select the chosen features from X_train and X_test
    X_train_selected = X_train[consensus_features]
    X_test_selected = X_test[consensus_features]

    return X_train_selected, X_test_selected, consensus_features


def save_processed_data(X_train, X_test, y_train, y_test, selected_features, output_dir):
    """
    Save processed data to CSV files

    Parameters:
    X_train, X_test (pd.DataFrame): Feature matrices
    y_train, y_test (pd.Series): Target variables
    selected_features (list): Selected feature names
    output_dir (str): Directory to save processed files
    """
    print(f"\nSaving processed data to {output_dir}...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save feature matrices
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)

    # Save labels
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    # Save selected feature names for reference
    with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    # Save a feature selection report
    report_path = os.path.join(output_dir, 'feature_engineering_report.md')
    with open(report_path, 'w') as f:
        f.write("# Feature Engineering Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Dataset Information\n")
        f.write(f"* Training samples: {X_train.shape[0]}\n")
        f.write(f"* Testing samples: {X_test.shape[0]}\n")
        f.write(f"* Selected features: {len(selected_features)}\n\n")

        f.write("## Selected Features\n")
        for i, feature in enumerate(selected_features, 1):
            f.write(f"{i}. {feature}\n")

        f.write("\n## Feature Statistics\n\n")
        f.write("| Feature | Mean | Std Dev | Min | Max |\n")
        f.write("|---------|------|---------|-----|-----|\n")

        # Calculate statistics on original (non-scaled) data if available
        for feature in selected_features:
            mean = X_train[feature].mean()
            std = X_train[feature].std()
            min_val = X_train[feature].min()
            max_val = X_train[feature].max()
            f.write(f"| {feature} | {mean:.3f} | {std:.3f} | {min_val:.3f} | {max_val:.3f} |\n")

    print("Data and reports saved successfully.")


def main():
    """Main function to orchestrate the feature engineering process"""
    print("\n" + "=" * 50)
    print("HEART DISEASE FEATURE ENGINEERING".center(50))
    print("=" * 50 + "\n")

    # Define paths
    input_path = '../data/processed/cleaned_data.csv'
    output_dir = '../data/processed'

    # Step 1: Load cleaned data
    df = load_cleaned_data(input_path)

    # Step 2: Create new engineered features
    df_engineered = create_new_features(df)

    # Step 3: Split data into train/test BEFORE any scaling/normalization
    X_train, X_test, y_train, y_test = split_data(df_engineered)

    # Step 4: Preprocess features (scaling)
    X_train_processed, X_test_processed, numerical_cols, scaler = preprocess_features(X_train, X_test)

    # Step 5: Select best features
    X_train_selected, X_test_selected, selected_features = select_features(
        X_train_processed, y_train, X_test_processed, n_features=15
    )

    # Step 6: Save processed data
    save_processed_data(X_train_selected, X_test_selected, y_train, y_test, selected_features, output_dir)

    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING COMPLETE".center(50))
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()