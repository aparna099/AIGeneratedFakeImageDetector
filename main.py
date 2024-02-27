from src import data_loader
from src import vgg_model
from src import lenet_model
from src import resnet_model
from src import densenet_model
# from src import vggface_model
from src import cnn_model
from src import evaluation
from src import utils
import tensorflow as tf

# Load images using the data loader
images_processed, labels_processed = data_loader.load_images_from_directory(utils.img_dir)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = data_loader.split_data(images_processed, labels_processed)


# lenet_model.train_lenet_model(x_train, y_train, utils.lenet_final_model_path)

# resnet_model.train_resnet_model(x_train, y_train, utils.resnet_final_model_path)

# densenet_model.train_densenet_model(x_train, y_train, utils.densenet_final_model_path)

# vggface_model.train_vggface_model(x_train, y_train, utils.vggface_final_model_path)

# # Train the VGG model
# vgg_model.train_vgg_model(x_train, y_train, utils.vgg_final_model_path)

# # Train the CNN model
# cnn_model.train_cnn_model(x_train, y_train, utils.cnn_final_model_path)

# Evaluate the model
model_loss, model_accuracy = evaluation.evaluate_model(x_test, y_test)

print("Model Accuracy:", model_accuracy)
print("Model Loss:", model_loss )
