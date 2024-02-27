import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_vggface import VGGFace
from src import utils

def build_vggface(input_shape=utils.input_shape, num_classes=utils.num_classes, dropout_rate=utils.dropout_rate):
    """
    Builds the VGGFace model with specified input shape, number of classes, and dropout rate.

    Args:
        input_shape (tuple): Input shape of the images.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.models.Model: Built VGGFace model.
    """
    base_model = VGGFace(model='vgg16', include_top=False, input_shape=input_shape, pooling='avg')

    # Load pre-trained weights
    loaded_model =  tf.keras.models.load_weights(utils.vgg_final_model_path)

    # Set the base model's weights
    base_model.set_weights(loaded_model.get_weights())

    # Freeze the base model's layers
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        Dense(4096, activation='relu'),
        Dropout(dropout_rate),
        Dense(4096, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_vggface_model(x_train, y_train_encoded, model_path):
    """
    Trains the VGGFace model.

    Args:
        x_train (numpy.ndarray): Input training data.
        y_train_encoded (numpy.ndarray): Encoded training labels.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Build the VGGFace model
    vggface_model = build_vggface()

    # Create the Adam optimizer
    optimizer = Adam(learning_rate=utils.learning_rate)

    print('Training VGGFace model...')

    # Compile the VGGFace model
    vggface_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the ModelCheckpoint callback for VGGFace model
    checkpoint_callback_vggface = ModelCheckpoint(utils.vggface_best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the VGGFace model
    history_vggface = vggface_model.fit(x_train, y_train_encoded, epochs=utils.vggface_epochs, batch_size=utils.batch_size, validation_split=utils.validation_split, callbacks=[checkpoint_callback_vggface])

    # Save the final VGGFace model
    vggface_model.save(model_path)
