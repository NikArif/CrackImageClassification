# %%
#1. Setup - importing packages
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, applications, models
import numpy as np
import matplotlib.pyplot as plt
import os, datetime
import cv2
import imghdr
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# %%
# get file path and read the file
PATH = os.getcwd()
data = os.path.join(PATH,'data')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

dataset = tf.keras.utils.image_dataset_from_directory(data, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
# %%
# Inspect data example
class_names = dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    plt.grid("off")
# %%
# Calculate the total number of batches in the dataset
dataset_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the data into train, val, and test datasets
train_dataset = dataset.take(train_size)
remaining_dataset = dataset.skip(train_size)
val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# %%
#Convert tensorflow dataset into PreFetch dataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
# %%
# Create a Sequential for image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %%
#7. Visualizing data augmentation
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

# %%
# Data Normalization - define  a layer for it
preprocess_input = applications.mobilenet_v2.preprocess_input
# %%
# Construct the transfer learning pipeline
# pipeline : data augmentation > preprocess input > transfer learning model
# load the pretrained model using keras.applications
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
base_model.summary()
keras.utils.plot_model(base_model)
# %%
# Freeze the entire feature extractor
base_model.trainable = False
base_model.summary()
# %%
# Create the global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
# Create output layer with Dense layer
output_layer = layers.Dense(len(class_names),activation='softmax')
# Build the entire pipeline using functional API
#a. Input 
inputs = keras.Input(shape=IMG_SHAPE)
#b. Data augmentation
x = data_augmentation(inputs)
#c. data normalization
x = preprocess_input(x)
#d. Transfer learning features extractor
x = base_model(x,training=False)
#e. classification layers
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)
#f. Build the model
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

# %%
# Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# %%
# Prepare the callback object for model.fit
early_stopping = callbacks.EarlyStopping(patience=2)
PATH = os.getcwd()
logpath = os.path.join(PATH,"tensorboard_log",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(logpath)

# %%
# Evaluate the model with test data before training
model.evaluate(test_dataset)
# %%
# model training
EPOCHS = 10
history = model.fit(
  train_dataset,
  validation_data = val_dataset,
  epochs=EPOCHS, callbacks=[early_stopping,tb]
)

# %%
# Evaluate model after training
model.evaluate(test_dataset)

# %%
# Observing model performance by plotting accuracy and loss
# Accuracy
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()
# %%
# Loss
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()
# %%
# Get the model architecture
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_architecture.png',show_shapes=True)

# %%
# Saving the model in .h5 format
from tensorflow.keras.models import load_model
model.save(os.path.join('image_classification.h5'))
# %%
# Model deployment
# Retrieve a batch of data from test data and perform prediction
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
# %%
# Identify the class for the predicition
prediction_indexes = np.argmax(predictions, axis=1)

# %%
# Display the result using matplotlib
label_map ={i:names for i,names in enumerate(class_names)}
prediction_list = [label_map[i] for i in prediction_indexes]
label_list = [label_map[i] for i in label_batch]
# %%
# plot the image graph using matplotlib
plt.figure(figsize=(20,20))
for i in range(9):
  ax = plt.subplot(3,3,i+1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(f"Predicition: {prediction_list[i]}, Label: {label_list[i]}")
  plt.axis("off")
  plt.grid("off")
# %%
