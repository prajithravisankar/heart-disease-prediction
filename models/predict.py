
def predict_heart_disease(patient_data):
    """
    Predict heart disease risk for a patient.

    Parameters:
        patient_data (dict): Patient health data

    Returns:
        dict: Prediction results with probability and risk level
    """
    import joblib
    import pandas as pd
    import os

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'xgboost_heart_disease_model.joblib')
    model = joblib.load(model_path)

    # Required features for the model
    required_features = [np.str_('Age'), np.str_('Cholesterol'), np.str_('Exercise Hours'), np.str_('Gender_Male'), np.str_('Smoking_Former'), np.str_('Smoking_Never'), np.str_('Alcohol Intake_Moderate'), np.str_('Family History_Yes'), np.str_('Chest Pain Type_Atypical Angina'), np.str_('Chest Pain Type_Non-anginal Pain')]

    # Create DataFrame with patient data
    df = pd.DataFrame([patient_data])

    # Check for missing features
    missing_features = set(required_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")

    # Select only the features used by the model
    df = df[required_features]

    # Make prediction
    probability = model.predict_proba(df)[0, 1]
    prediction = int(probability >= 0.5)

    # Determine risk level
    if probability < 0.2:
        risk_level = "Low"
    elif probability < 0.5:
        risk_level = "Moderate"
    elif probability < 0.8:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return {
        "prediction": prediction,
        "probability": float(probability),
        "risk_level": risk_level
    }
