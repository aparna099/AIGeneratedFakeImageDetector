import tensorflow as tf
from src import utils

def evaluate_model(x_test, y_test_encoded):
    """
    Evaluates the trained model on the test data.

    Args:
        x_test (numpy.ndarray): Input test data.
        y_test (numpy.ndarray): Test labels.

    Returns:
        tuple: Loss and accuracy of the model.
    """
    print('Evaluating the model...')

    # Load the trained model
    model = tf.keras.models.load_model(utils.final_model_path)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test_encoded)

    return loss, accuracy
