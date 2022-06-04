# ******* Imports **********
import numpy as np
import cv2
import os
import math
import pylab
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import splitfolders

np.random.seed(42)
tf.random.set_seed(42)

# Split Images to train, validation, and test folders
input_folder = "brain directory"
output_folder = "out put directory"
splitfolders.ratio(input_folder,
                   output=output_folder,
                   seed=42,
                   ratio=(0.7, 0.2, 0.1),
                   group_prefix=None)  # default values

# Image's Size
BATCH_SIZE = 32
IMG_SIZE = (299, 299)

# Generate Datasets Using tf.keras.utils.image_dataset_from_directory

train_dir = "train directory"
validation_dir = "val directory"
test_dir = "test directory"

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                           shuffle=True,
                                                           batch_size=BATCH_SIZE,
                                                           image_size=IMG_SIZE)

# Print batches size
print("Number of validation batches: %d" % tf.data.experimental.cardinality(validation_dataset))
print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

# Show class names
class_names = train_dataset.class_names
print(class_names)

# Show 9 images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# Auto Tune Dataset for better result
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Pre Process datasets for Inceptionv3
preprocess_input = tf.keras.applications.inception_v3.preprocess_input

# Image Shape
IMG_SHAPE = IMG_SIZE + (3,)

# Download inception model and it's weights
base_model = tf.keras.applications.inception_v3.InceptionV3(
    weights="imagenet",
    include_top=False)

# Print base model shape
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

np.random.seed(42)
tf.random.set_seed(42)

# Create Top layers for better brain image features recognition
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")

inputs = tf.keras.Input(shape=(299, 299, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Freeze Lower layers to prevent lower layer's weight update
for layer in base_model.layers:
    layer.trainable = False

# Compile the Model
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

# Print loss0, accuracy0
initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Train freeze model
# epochs = 10 >>>> Just Learning Top Layers
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=validation_dataset)

# Show accuracy, val_accuracy, loss, val_loss plots
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# unfreeze lower layers
for layer in base_model.layers:
    layer.trainable = True

# Optimize model
# Using early stopping and model checkpoint to save the best model by monitoring the val accuracy

model_checkpoint_folder = "directory" + "model name" + ".h5"
optimizer = keras.optimizers.Adam(learning_rate=0.001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_folder, save_best_only=True)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=optimizer,
              metrics=["accuracy"])
# print model's summary
print(model.summary())

# Train model for 200 epochs
history = model.fit(train_dataset,
                    epochs=200,
                    validation_data=validation_dataset,
                    callbacks=[early_stopping, model_checkpoint])

# print model's accuracy on test dataset
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)
