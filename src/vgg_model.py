import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src import utils

def build_vgg19(input_shape=utils.input_shape, num_classes=utils.num_classes, dropout_rate=utils.dropout_rate):
    """
    Builds the VGG19 model with specified input shape, number of classes, and dropout rate.

    Args:
        input_shape (tuple): Input shape of the images.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.models.Model: Built VGG19 model.
    """
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def train_vgg_model(x_train, y_train_encoded, model_path):
    """
    Trains the VGG model with transfer learning.

    Args:
        x_train (numpy.ndarray): Input training data.
        y_train_encoded (numpy.ndarray): Encoded training labels.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Build the VGG19 model
    vgg_model = build_vgg19()

    # Create the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=utils.learning_rate,
        beta_1=utils.momentum,
        beta_2=0.999,
        epsilon=1e-07
    )

    print('Training VGG model...')

    # Compile the VGG model
    vgg_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the ModelCheckpoint callback for VGG model
    checkpoint_callback_vgg = ModelCheckpoint(utils.vgg_best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the VGG model with transfer learning
    history_vgg = vgg_model.fit(x_train, y_train_encoded, epochs=utils.vgg_epochs, batch_size=utils.batch_size, validation_split=utils.validation_split, callbacks=[checkpoint_callback_vgg])

    # Save the final vgg model
    vgg_model.save(model_path)
