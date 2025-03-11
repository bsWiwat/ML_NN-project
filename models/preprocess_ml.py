import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_heart_disease_data():
    """
    Load the Heart Disease dataset from UCI repository
    """
    try:
        # Load data (you would need to download this from UCI repository)
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        # data = pd.read_csv('../data/heart.csv')
        data = pd.read_csv('heart.csv')
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data):
    """
    Preprocess the heart disease data
    """
    try:
        # Select features we'll use for the demo
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
        X = data[features]
        y = data['target']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save the scaler
        joblib.dump(scaler, 'scaler.pkl')
        logger.info("Scaler saved successfully")

        return X_train_scaled, X_test_scaled, y_train, y_test

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load and preprocess data
        data = load_heart_disease_data()
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(data)
        logger.info("Data preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
