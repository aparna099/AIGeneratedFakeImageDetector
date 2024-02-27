import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src import utils

def build_lenet(input_shape=utils.input_shape, num_classes=utils.num_classes, dropout_rate=utils.dropout_rate):
    """
    Builds the LeNet model with specified input shape, number of classes, and dropout rate.

    Args:
        input_shape (tuple): Input shape of the images.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.models.Model: Built LeNet model.
    """
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_lenet_model(x_train, y_train_encoded, model_path):
    """
    Trains the LeNet model.

    Args:
        x_train (numpy.ndarray): Input training data.
        y_train_encoded (numpy.ndarray): Encoded training labels.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Build the LeNet model
    lenet_model = build_lenet()

    # Create the Adam optimizer
    optimizer = Adam(learning_rate=utils.learning_rate)

    print('Training LeNet model...')

    # Compile the LeNet model
    lenet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the ModelCheckpoint callback for LeNet model
    checkpoint_callback_lenet = ModelCheckpoint(utils.lenet_best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the LeNet model
    history_lenet = lenet_model.fit(x_train, y_train_encoded, epochs=utils.lenet_epochs, batch_size=utils.batch_size, validation_split=utils.validation_split, callbacks=[checkpoint_callback_lenet])

    # Save the final LeNet model
    lenet_model.save(model_path)
