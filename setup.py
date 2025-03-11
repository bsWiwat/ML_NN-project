import os
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """
    Check and install required packages
    """
    try:
        logger.info("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Required packages installed successfully")
    except Exception as e:
        logger.error(f"Error installing requirements: {str(e)}")
        raise

def setup_directories():
    """
    Create necessary directories
    """
    try:
        directories = ['data', 'models', 'uploads']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise

def train_models():
    """
    Train both ML and NN models
    """
    try:
        # Train ML model
        logger.info("Training Machine Learning model...")
        subprocess.check_call([sys.executable, "models/train_ml_model.py"])
        logger.info("ML model trained successfully")

        # Train NN model
        logger.info("Training Neural Network model...")
        subprocess.check_call([sys.executable, "models/train_nn_model.py"])
        logger.info("NN model trained successfully")
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

def main():
    """
    Main setup function
    """
    try:
        logger.info("Starting setup process...")
        
        # Check and install requirements
        check_requirements()
        
        # Create necessary directories
        setup_directories()
        
        # Train models
        train_models()
        
        logger.info(" Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        logger.error("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
