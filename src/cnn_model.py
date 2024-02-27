import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src import utils

def create_cnn_model(input_shape):
    """
    Creates a CNN model with VGG architecture.

    Args:
        input_shape (tuple): Input shape of the images.

    Returns:
        tf.keras.models.Model: Built CNN model.
    """
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='dense'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dropout1'))
    model.add(Dropout(0.5))
    model.add(Dense(utils.num_classes, activation='softmax', name='predictions'))

    return model

def train_cnn_model(x_train, y_train_encoded, model_path):
    """
    Trains the CNN model with transfer learning using pre-trained VGG weights.

    Args:
        x_train (numpy.ndarray): Input training data.
        y_train_encoded (numpy.ndarray): Encoded training labels.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Create the CNN model
    cnn_model = create_cnn_model(input_shape=utils.input_shape)

    # Load the saved VGG model weights
    loaded_model = tf.keras.models.load_model(utils.vgg_best_model_path)

    # Load the weights from the trained VGG model to the CNN model
    cnn_model.set_weights(loaded_model.get_weights())

    # Create the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=utils.learning_rate,
        beta_1=utils.momentum,
        beta_2=0.999,
        epsilon=1e-07
    )

    print('Training CNN model...')

    # Compile the model
    cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(utils.cnn_best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the CNN model
    history_cnn = cnn_model.fit(x_train, y_train_encoded, epochs=utils.cnn_epochs, batch_size=utils.batch_size, validation_split=0.2, callbacks=[checkpoint_callback])

    # Save the final cnn model and final model
    cnn_model.save(model_path)
    cnn_model.save(utils.final_model_path)