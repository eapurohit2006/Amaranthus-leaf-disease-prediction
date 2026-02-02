import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

# 1. Load trained model
model = tf.keras.models.load_model(r"C:\Users\rohit\minor_web\amaranthus_resnet50v2.keras")


# 2. Load class indices
with open("class_indices_resnet50v2.json", "r") as f:
    class_indices = json.load(f)
    
# Reverse dictionary {0: "Healthy", 1: "..."}
class_names = {v: k for k, v in class_indices.items()}

# 3. Take image path as user input
img_path = input("Enter the path to the image you want to classify: ").strip()

# 4. Preprocess image
try:
    img = image.load_img(img_path, target_size=(224, 224))
except FileNotFoundError:
    print(f"❌ Error: File not found at '{img_path}'")
    exit()

img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 5. Predict
predictions = model.predict(img_array)
pred_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print(f"\n✅ Predicted Class: {class_names[pred_class]}")
print(f"🔹 Confidence: {confidence:.2f}%")

# Print all class probabilities
print("\nAll class probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"  {class_names[i]}: {prob*100:.2f}%")