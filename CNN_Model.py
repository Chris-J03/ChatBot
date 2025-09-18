import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import kerastuner as kt

def convert_images_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpeg', '.webp', '.tiff')):  # Adjust for other formats if needed
            img_path = os.path.join(directory, filename)
            try:
                with Image.open(img_path) as img:
                    new_img_path = os.path.join(directory, filename.rsplit('.', 1)[0] + '.jpg')
                    img.convert('RGB').save(new_img_path, 'JPEG')
                    os.remove(img_path)  # Optionally, remove original file
                    print(f"Converted {filename} to JPG.")
            except Exception as e:
                print(f"Could not process {filename}: {e}")

# Convert all images in both directories to jpg format
convert_images_to_jpg("cod_images/Captain_Price")
convert_images_to_jpg("cod_images/Woods")

# Remove corrupted images
def remove_corrupted_images(directory):
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify the integrity of the image
        except (IOError, SyntaxError):
            print(f"Corrupted image removed: {filename}")
            os.remove(img_path)

remove_corrupted_images("cod_images/Captain_Price")
remove_corrupted_images("cod_images/Woods")

# Function to load images and skip invalid files
def load_and_process_image(image_path):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)  # Decode as JPEG
        img = tf.image.resize(img, [64, 64])  # Resize to 128x128
        img = img / 255.0  # Normalize to [0, 1]
        return img
    except Exception as e:
        print(f"Skipping invalid image: {image_path} due to {e}")
        return None

# Custom generator to load images from directory and skip invalid ones
def image_generator(directory):
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = load_and_process_image(img_path)
                if img is not None:
                    yield img, folder_name

# Load dataset with the custom generator
def create_dataset(directory):
    image_list = []
    label_list = []
    class_names = os.listdir(directory)  # Folder names are the class labels

    for img, label in image_generator(directory):
        image_list.append(img)
        label_list.append(class_names.index(label))  # Convert label to index

    # Convert lists to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
    return dataset

train_dataset = create_dataset("cod_images")

# Print class names to confirm that TensorFlow recognizes the two classes
print("Classes detected:", ['Captain_Price', 'Woods'])

# Check class distribution in dataset
def check_class_distribution(directory):
    class_counts = {}
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            class_counts[folder_name] = len(os.listdir(folder_path))
    
    # Plot the class distribution
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.show()

# Define the original model
def build_original_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')  # 2 classes
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# Define the model-building function for hyperparameter tuning
def build_model(hp):
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters=hp.Int('conv1_filter', min_value=32, max_value=64, step=32),
            kernel_size=3,
            activation='relu',
            input_shape=(64, 64, 3)
        ),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(
            filters=hp.Int('conv2_filter', min_value=64, max_value=128, step=64),
            kernel_size=3,
            activation='relu'
        ),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=hp.Int('dense_units', min_value=64, max_value=128, step=64),
            activation='relu'
        ),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4])),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# Split the dataset into training and validation sets
train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
train_size = int(0.8 * len(train_dataset))
train_data = train_dataset.take(train_size).batch(32).cache()
val_data = train_dataset.skip(train_size).batch(32).cache()

# Train and evaluate the original model
class_weights = {0: 1.0, 1: 1.0}  # Example: Class 1 has 3x more weight than Class 0
original_model = build_original_model()
original_model.fit(train_data, epochs=10, validation_data=val_data, class_weight=class_weights)
original_test_loss, original_test_acc = original_model.evaluate(val_data, verbose=2)
print('\nOriginal Model Test Accuracy:', original_test_acc)

# Save the original model
original_model.save("cod_character_classifier.h5")
print("Original model saved as 'cod_character_classifier.h5'.")

# Initialize the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Limit the number of trials
    directory='my_tuning_dir',
    project_name='cod_character_classifier'
)

# Perform hyperparameter tuning
tuner.search(train_data, epochs=10, validation_data=val_data, class_weight=class_weights)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and train the best model
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(train_data, epochs=10, validation_data=val_data, class_weight=class_weights)
best_test_loss, best_test_acc = best_model.evaluate(val_data, verbose=2)
print('\nBest Model Test Accuracy:', best_test_acc)

# Save the best model
best_model.save("best_cod_character_classifier.h5")
print("Best model saved as 'best_cod_character_classifier.h5'.")

# Compare the performance of the two models
print("\nModel Comparison:")
print(f"Original Model Test Accuracy: {original_test_acc:.4f}")
print(f"Best Model Test Accuracy: {best_test_acc:.4f}")