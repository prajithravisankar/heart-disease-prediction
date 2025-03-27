import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import os


def load_cleaned_data(input_path):
    """
    Load the cleaned dataset

    Parameters:
    input_path (str): Path to the cleaned CSV file

    Returns:
    pd.DataFrame: Cleaned dataset
    """
    return pd.read_csv(input_path)


def engineer_features(df):
    """
    Perform feature engineering

    Parameters:
    df (pd.DataFrame): Input dataframe

    Returns:
    tuple: X (features), y (target), feature_names
    """
    # Separate features and target
    y = df['Heart Disease']
    X = df.drop('Heart Disease', axis=1)

    # Ensure all columns are numeric
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Encode categorical columns
    le = LabelEncoder()
    for col in categorical_columns:
        X[col] = le.fit_transform(X[col].astype(str))

    # Encode target variable (binary classification)
    y = pd.Series(le.fit_transform(y.astype(str)), name='Heart Disease')

    # Store original feature names
    feature_names = X.columns.tolist()

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=8)  # Select top 8 features
    X_selected = selector.fit_transform(X_scaled, y)

    # Get selected feature names
    selected_feature_mask = selector.get_support()
    selected_features = [feature for feature, selected in zip(feature_names, selected_feature_mask) if selected]

    print("Selected Features:")
    for idx, feature in enumerate(selected_features, 1):
        print(f"{idx}. {feature}")

    return X_selected, y, selected_features


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets

    Parameters:
    X (array-like): Features
    y (array-like): Target variable
    test_size (float): Proportion of test set
    random_state (int): Random seed for reproducibility

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Save processed data to CSV files

    Parameters:
    X_train, X_test, y_train, y_test (array-like): Train and test data
    output_dir (str): Directory to save processed files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Verify and trim to match feature matrix shape
    min_length = min(len(X_train), len(y_train))
    X_train = X_train[:min_length]
    y_train = y_train[:min_length]

    min_length = min(len(X_test), len(y_test))
    X_test = X_test[:min_length]
    y_test = y_test[:min_length]

    # Create feature DataFrames
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    # Save processed data
    X_train_df.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    pd.Series(y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    pd.Series(y_test).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    print("Processed data saved successfully.")
    print("X_train shape:", X_train_df.shape)
    print("y_train shape:", len(y_train))
    print("X_test shape:", X_test_df.shape)
    print("y_test shape:", len(y_test))

def main():
    # Paths
    input_path = '/Users/prajithravisankar/Documents/lakehead/bigData/project/heart-disease-prediction/data/processed/cleaned_data.csv'
    output_dir = '/Users/prajithravisankar/Documents/lakehead/bigData/project/heart-disease-prediction/data/processed'

    # Load cleaned data
    df = load_cleaned_data(input_path)

    # Engineer features
    X, y, selected_features = engineer_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, output_dir)


if __name__ == '__main__':
    main()