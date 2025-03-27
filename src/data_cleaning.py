import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(input_path):
    """
    Load and clean the heart disease dataset

    Parameters:
    input_path (str): Path to the input CSV file

    Returns:
    pd.DataFrame: Cleaned dataset
    """
    # Load the dataset
    df = pd.read_csv(input_path)

    # Check for duplicates
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)

    # Identify missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum())

    # Handle categorical variable encoding
    categorical_columns = ['Gender', 'Smoking', 'Alcohol Intake', 'Family History',
                           'Diabetes', 'Obesity', 'Exercise Induced Angina',
                           'Chest Pain Type']

    # Label Encoding for categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].fillna('Unknown').astype(str))

    # Handle missing values using KNNImputer
    # Separate numeric columns for imputation
    numeric_columns = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate',
                       'Exercise Hours', 'Stress Level', 'Blood Sugar']

    # Prepare data for imputation
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Detect and handle outliers using IQR method
    def remove_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Print outlier statistics
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"Outliers in {col}: {len(outliers)} rows")

            # Clip outliers to bounds
            df[col] = np.clip(df[col], lower_bound, upper_bound)

        return df

    # Remove outliers from numeric columns
    df = remove_outliers(df, numeric_columns)

    # Final missing value check
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df


def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataset to a CSV file

    Parameters:
    df (pd.DataFrame): Cleaned dataset
    output_path (str): Path to save the cleaned CSV
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


def main():
    # Paths for input and output
    input_path = '/Users/prajithravisankar/Documents/lakehead/bigData/project/heart-disease-prediction/data/raw/project 2.csv'
    output_path = '/Users/prajithravisankar/Documents/lakehead/bigData/project/heart-disease-prediction/data/processed/cleaned_data.csv'

    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Clean the data
    cleaned_df = load_and_clean_data(input_path)

    # Save the cleaned data
    save_cleaned_data(cleaned_df, output_path)


if __name__ == '__main__':
    main()