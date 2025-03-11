import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """
    Load and preprocess Fashion MNIST dataset
    """
    try:
        # Load Fashion MNIST dataset
        logger.info("Loading Fashion MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        # Normalize pixel values
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # Reshape for CNN (add channel dimension)
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        logger.info("Data preprocessing completed successfully")
        return (X_train, y_train), (X_test, y_test)

    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def create_model():
    """
    Create and compile the CNN model
    """
    try:
        model = Sequential([
            # First Convolutional Layer
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Layer
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Layer
            Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten layer
            Flatten(),
            
            # Dense layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Model created successfully")
        model.summary()
        return model

    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train the CNN model
    """
    try:
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train the model
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping]
        )

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")

        # Save the model
        model.save('fashion_mnist_model.h5')
        logger.info("Model saved successfully")

        return history

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    """
    try:
        # Make predictions on test set
        predictions = model.predict(X_test)
        
        # Calculate accuracy per class
        y_test_classes = np.argmax(y_test, axis=1)
        pred_classes = np.argmax(predictions, axis=1)
        
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # Print accuracy for each class
        logger.info("\nPer-class Accuracy:")
        for i in range(10):
            mask = y_test_classes == i
            class_accuracy = np.mean(pred_classes[mask] == y_test_classes[mask])
            logger.info(f"{class_names[i]}: {class_accuracy:.4f}")

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load and preprocess data
        (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()

        # Create model
        model = create_model()

        # Train model
        history = train_model(model, X_train, y_train, X_test, y_test)

        # Evaluate model
        evaluate_model(model, X_test, y_test)

        logger.info("Model training and evaluation completed successfully")

    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
