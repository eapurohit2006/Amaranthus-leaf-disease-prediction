import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset
data_dir = data_dir = r"C:\Users\rohit\OneDrive\Desktop\dataset"


# Preprocessing & Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Train data
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # resize images
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Check classes
print(train_data.class_indices)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Load MobileNetV2 as base model (pre-trained on ImageNet)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base layers

# Build model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(train_data.class_indices), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# Save model
model.save("amaranthus_mobilenetv2.keras")
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Model Loss")
plt.show()

import json

# Save the class labels (mapping of folder names to numbers)
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
import json  

# Save class indices
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
