import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src import utils

def build_resnet(input_shape=utils.input_shape, num_classes=utils.num_classes, dropout_rate=utils.dropout_rate):
    """
    Builds the ResNet model with specified input shape, number of classes, and dropout rate.

    Args:
        input_shape (tuple): Input shape of the images.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.models.Model: Built ResNet model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = resnet_block(x, filters=[64, 64], strides=(1, 1))
    x = resnet_block(x, filters=[64, 64])
    x = resnet_block(x, filters=[128, 128], strides=(2, 2))
    x = resnet_block(x, filters=[128, 128])
    x = resnet_block(x, filters=[256, 256], strides=(2, 2))
    x = resnet_block(x, filters=[256, 256])
    x = resnet_block(x, filters=[512, 512], strides=(2, 2))
    x = resnet_block(x, filters=[512, 512])

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def resnet_block(x, filters, strides=(1, 1)):
    """
    Builds a single ResNet block with two convolutional layers and a shortcut connection.

    Args:
        x (tf.Tensor): Input tensor.
        filters (list): List of integers specifying the number of filters in each convolutional layer.
        strides (tuple): Strides for the convolutional layers.

    Returns:
        tf.Tensor: Output tensor.
    """
    shortcut = x
    x = Conv2D(filters=filters[0], kernel_size=(3, 3), strides=strides, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)

    if strides != (1, 1) or shortcut.shape[-1] != filters[1]:
        shortcut = Conv2D(filters=filters[1], kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.nn.relu(x)
    return x

def train_resnet_model(x_train, y_train_encoded, model_path):
    """
    Trains the ResNet model.

    Args:
        x_train (numpy.ndarray): Input training data.
        y_train_encoded (numpy.ndarray): Encoded training labels.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Build the ResNet model
    resnet_model = build_resnet()

    # Create the Adam optimizer
    optimizer = Adam(learning_rate=utils.learning_rate)

    print('Training ResNet model...')

    # Compile the ResNet model
    resnet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the ModelCheckpoint callback for ResNet model
    checkpoint_callback_resnet = ModelCheckpoint(utils.resnet_best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the ResNet model
    history_resnet = resnet_model.fit(x_train, y_train_encoded, epochs=utils.resnet_epochs, batch_size=utils.batch_size, validation_split=utils.validation_split, callbacks=[checkpoint_callback_resnet])

    # Save the final ResNet model
    resnet_model.save(model_path)
