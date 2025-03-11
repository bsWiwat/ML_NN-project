import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from preprocess_ml import load_heart_disease_data, preprocess_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model for heart disease prediction
    """
    try:
        # Initialize the model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # Train the model
        logger.info("Training Random Forest model...")
        model.fit(X_train, y_train)

        # Make predictions on test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.4f}")

        # Print detailed classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")

        # Save the model
        joblib.dump(model, 'heart_disease_model.pkl')
        logger.info("Model saved successfully")

        return model

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info("Model Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Print feature importances
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            logger.info(f"Feature {name}: {importance:.4f}")

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = load_heart_disease_data()
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(data)

        # Train model
        logger.info("Training model...")
        model = train_model(X_train_scaled, X_test_scaled, y_train, y_test)

        # Evaluate model
        logger.info("Evaluating model...")
        evaluate_model(model, X_test_scaled, y_test)

        logger.info("Model training and evaluation completed successfully")

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
