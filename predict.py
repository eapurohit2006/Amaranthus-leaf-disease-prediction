import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

# 1. Load trained model
model = tf.keras.models.load_model("amaranthus_mobilenetv2.keras")

# 2. Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
# Reverse dictionary {0: "Healthy", 1: "..."}
class_names = {v: k for k, v in class_indices.items()}

# 3. Path to your test image
img_path = r"C:\Users\rohit\OneDrive\文档\test_leaf.jpeg"

# 4. Preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 5. Predict
predictions = model.predict(img_array)
pred_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print(f"Predicted Class: {class_names[pred_class]}")
print(f"Confidence: {confidence:.2f}%")
