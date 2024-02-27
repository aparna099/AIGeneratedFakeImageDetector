import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src import utils

def build_densenet(input_shape=utils.input_shape, num_classes=utils.num_classes, dropout_rate=utils.dropout_rate):
    """
    Builds the DenseNet121 model with specified input shape, number of classes, and dropout rate.

    Args:
        input_shape (tuple): Input shape of the images.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.models.Model: Built DenseNet121 model.
    """
    base_model = DenseNet121(include_top=False, input_shape=input_shape)

    # Freeze the base model's layers
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_densenet_model(x_train, y_train_encoded, model_path):
    """
    Trains the DenseNet121 model.

    Args:
        x_train (numpy.ndarray): Input training data.
        y_train_encoded (numpy.ndarray): Encoded training labels.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Build the DenseNet121 model
    densenet_model = build_densenet()

    # Create the Adam optimizer
    optimizer = Adam(learning_rate=utils.learning_rate)

    print('Training DenseNet121 model...')

    # Compile the DenseNet121 model
    densenet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the ModelCheckpoint callback for DenseNet121 model
    checkpoint_callback_densenet = ModelCheckpoint(utils.densenet_best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the DenseNet121 model
    history_densenet = densenet_model.fit(x_train, y_train_encoded, epochs=utils.densenet_epochs, batch_size=utils.batch_size, validation_split=utils.validation_split, callbacks=[checkpoint_callback_densenet])

    # Save the final DenseNet121 model
    densenet_model.save(model_path)
