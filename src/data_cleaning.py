import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder


def load_and_clean_data(input_path):
    """
    Load and clean the heart disease dataset with improved handling
    of missing values, categorical features, and outliers.

    Parameters:
    input_path (str): Path to the input CSV file

    Returns:
    pd.DataFrame: Cleaned dataset
    """
    # Load the dataset
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")

    # Display initial data summary
    print("\nInitial data summary:")
    print(df.describe().T)

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Dropped {duplicates} duplicate rows")

    # Identify missing values
    missing_values = df.isnull().sum()
    print("\nMissing values before cleaning:")
    print(missing_values)

    # Separate numerical and categorical columns for appropriate handling
    numerical_cols = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate',
                      'Exercise Hours', 'Stress Level', 'Blood Sugar']
    categorical_cols = ['Gender', 'Smoking', 'Alcohol Intake', 'Family History',
                        'Diabetes', 'Obesity', 'Exercise Induced Angina',
                        'Chest Pain Type']

    # Print value distributions for categorical columns
    print("\nCategorical column distributions:")
    for col in categorical_cols:
        if not df[col].isnull().all():  # Skip if all values are null
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False))

    # Step 1: Handle missing values - use most frequent for categorical data first
    for col in categorical_cols:
        if df[col].isnull().any():
            most_frequent = df[col].mode()[0]
            missing_count = df[col].isnull().sum()
            print(f"Filling {missing_count} missing values in '{col}' with most frequent value: '{most_frequent}'")
            # Avoid inplace operation that triggers pandas warning
            df[col] = df[col].fillna(most_frequent)

    # Step 2: Handle numerical missing values using KNNImputer
    # This is done AFTER categorical imputation to allow KNN to use categorical features
    if df[numerical_cols].isnull().any().any():
        print("\nImputing missing numerical values with KNNImputer...")

        # Apply KNNImputer only to numerical columns
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        # Extract just the numerical columns for imputation
        numerical_df = df[numerical_cols].copy()
        # Impute missing values
        imputed_numerical = imputer.fit_transform(numerical_df)

        # Convert back to DataFrame and assign to original dataframe
        imputed_df = pd.DataFrame(imputed_numerical, columns=numerical_cols, index=df.index)
        for col in numerical_cols:
            df[col] = imputed_df[col]

        print("Numerical imputation complete.")

    # Step 3: Check for and handle outliers using IQR method
    print("\nIdentifying and handling outliers...")
    outlier_counts = {}

    for col in numerical_cols:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)

        if len(outliers) > 0:
            # Clip outliers (less aggressive than removing rows)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    print("Outlier counts by column:")
    for col, count in outlier_counts.items():
        print(f"  {col}: {count} outliers {'(clipped to IQR bounds)' if count > 0 else ''}")

    # Step 4: One-hot encode categorical variables
    print("\nOne-hot encoding categorical variables...")

    # Create encoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity

    # Fit and transform
    encoded_cats = encoder.fit_transform(df[categorical_cols])

    # Get feature names
    feature_names = encoder.get_feature_names_out(categorical_cols)

    # Create DataFrame with encoded data
    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)

    # Remove original categorical columns and join with encoded columns
    df_final = df.drop(columns=categorical_cols).join(encoded_df)

    # Final check for missing values
    missing_after = df_final.isnull().sum().sum()
    print(f"\nTotal missing values after cleaning: {missing_after}")

    # Print shape after processing
    print(f"Final dataset shape: {df_final.shape[0]} rows, {df_final.shape[1]} columns")

    # Ensure target column is kept as 0/1
    print("\nTarget variable distribution:")
    print(df_final['Heart Disease'].value_counts())

    return df_final


def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataset to a CSV file

    Parameters:
    df (pd.DataFrame): Cleaned dataset
    output_path (str): Path to save the cleaned CSV
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to {output_path}")

    # Generate a data report
    generate_data_report(df, os.path.dirname(output_path))


def generate_data_report(df, output_dir):
    """
    Generate a simple report about the cleaned data

    Parameters:
    df (pd.DataFrame): Cleaned dataset
    output_dir (str): Directory to save the report
    """
    # Create a markdown report
    report = [
        "# Heart Disease Dataset Cleaning Report",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Summary",
        f"* Number of samples: {df.shape[0]}",
        f"* Number of features: {df.shape[1] - 1}",  # Excluding target column
        f"* Heart Disease cases: {df['Heart Disease'].sum()} ({df['Heart Disease'].mean() * 100:.2f}%)",
        "",
        "## Numeric Features Summary",
        "",
    ]

    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    report.append("| Feature | Min | Max | Mean | Median | Std Dev |")
    report.append("|---------|-----|-----|------|--------|---------|")

    for col in numeric_cols:
        # Skip non-numeric features and target
        if col == 'Heart Disease' or df[col].dtype not in ['int64', 'float64']:
            continue

        report.append(
            f"| {col} | {df[col].min():.2f} | {df[col].max():.2f} | {df[col].mean():.2f} | {df[col].median():.2f} | {df[col].std():.2f} |")

    # Add categorical features summary
    cat_features = [col for col in df.columns if
                    col.startswith(tuple(['Gender_', 'Smoking_', 'Alcohol Intake_', 'Family History_',
                                          'Diabetes_', 'Obesity_', 'Exercise Induced Angina_', 'Chest Pain Type_']))]

    if cat_features:
        report.append("")
        report.append("## Categorical Features (One-Hot Encoded)")
        report.append("")
        report.append("| Feature | Frequency | Percentage |")
        report.append("|---------|-----------|------------|")

        for col in cat_features:
            count = df[col].sum()
            percentage = (count / len(df)) * 100
            report.append(f"| {col} | {count} | {percentage:.2f}% |")

    # Write report to file
    report_path = os.path.join(output_dir, "data_cleaning_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Data cleaning report saved to {report_path}")


def main():
    # Define paths with correct relative paths from src directory
    input_path = '../data/raw/project 2.csv'
    output_path = '../data/processed/cleaned_data.csv'

    print("\n" + "=" * 50)
    print("HEART DISEASE DATASET CLEANING".center(50))
    print("=" * 50 + "\n")

    # Clean the data
    cleaned_df = load_and_clean_data(input_path)

    # Save the cleaned data
    save_cleaned_data(cleaned_df, output_path)

    print("\n" + "=" * 50)
    print("CLEANING PROCESS COMPLETE".center(50))
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()