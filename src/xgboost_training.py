import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GridSearchCV


def load_processed_data(data_dir):
    """
    Load processed training and testing data

    Parameters:
    data_dir (str): Directory containing processed data files

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    # Ensure files exist
    required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))

    # Read y_train and y_test, ensuring they are converted to numeric
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), header=None)

    # Ensure y_train and y_test are 1D arrays with correct length
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]

    # Verify and trim to match feature matrix shape
    min_length = min(len(X_train), len(y_train))
    X_train = X_train.iloc[:min_length]
    y_train = y_train.iloc[:min_length]

    # Verify and trim test data similarly
    min_length = min(len(X_test), len(y_test))
    X_test = X_test.iloc[:min_length]
    y_test = y_test.iloc[:min_length]

    # Convert to numeric, handling potential string representations
    y_train = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int)
    y_test = pd.to_numeric(y_test, errors='coerce').fillna(0).astype(int)

    # Verify dimensions and values
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Unique values in y_train:", y_train.unique())
    print("Unique values in y_test:", y_test.unique())

    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train):
    """
    Train XGBoost model with hyperparameter tuning

    Parameters:
    X_train (DataFrame): Training features
    y_train (Series): Training target

    Returns:
    xgb.XGBClassifier: Trained XGBoost model
    """
    # Define parameter grid for GridSearchCV
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Initialize base XGBoost classifier
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_estimators=100
    )

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Print best parameters
    print("Best Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, results_dir):
    """
    Evaluate model performance and generate visualizations

    Parameters:
    model (xgb.XGBClassifier): Trained XGBoost model
    X_test (DataFrame): Testing features
    y_test (Series): Testing target
    results_dir (str): Directory to save results
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate performance metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }

    # Print metrics
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

    # Feature Importance
    feature_importance = model.feature_importances_
    feature_names = X_test.columns

    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close()

    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df.to_csv(os.path.join(results_dir, 'model_metrics.csv'))

    return metrics


def main():
    # Paths
    data_dir = '/Users/prajithravisankar/Documents/lakehead/bigData/project/heart-disease-prediction/data/processed'
    results_dir = '/Users/prajithravisankar/Documents/lakehead/bigData/project/heart-disease-prediction/results'

    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    # Train XGBoost model with hyperparameter tuning
    xgboost_model = train_xgboost(X_train, y_train)

    # Evaluate model and generate results
    metrics = evaluate_model(xgboost_model, X_test, y_test, results_dir)


if __name__ == '__main__':
    main()

    # here