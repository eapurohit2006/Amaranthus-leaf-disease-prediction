"""
Model inference module - Uses the same prediction logic as predict1.py
This module is automatically loaded by backend/model_loader.py
"""
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
import io

# Load model and class indices on module import (lazy loading)
_model = None
_class_names = None


def _load_model():
	"""Load the model and class indices (singleton pattern)"""
	global _model, _class_names
	
	if _model is not None:
		return _model, _class_names
	
	# Get the project root directory (parent of 'model' directory)
	# When loaded from backend, we need to go up one level
	current_dir = Path(__file__).parent.parent.absolute()
	
	# 1. Load trained model (same as predict1.py)
	model_path = current_dir / "amaranthus_resnet50v2.keras"
	if not model_path.exists():
		# Try alternative path
		model_path = current_dir / "amaranthus_resnet50v2_high_acc.keras"
	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found. Looked in: {current_dir}")
	
	_model = tf.keras.models.load_model(str(model_path))
	
	# 2. Load class indices (same as predict1.py)
	cls_json_path = current_dir / "class_indices_resnet50v2.json"
	if not cls_json_path.exists():
		raise FileNotFoundError(f"Class indices file not found. Looked in: {current_dir}")
	
	with open(cls_json_path, "r") as f:
		class_indices = json.load(f)
	
	# Reverse dictionary {0: "Healthy", 1: "..."} - same as predict1.py
	_class_names = {v: k for k, v in class_indices.items()}
	
	return _model, _class_names


def predict(pil_image: Image.Image) -> list[dict[str, float]]:
	"""
	Predict using the same logic as predict1.py
	
	Args:
		pil_image: PIL Image object (received from FastAPI)
	
	Returns:
		list of dict with 'label' and 'probability' keys, sorted by probability (highest first)
	"""
	try:
		# Load model if not already loaded
		model, class_names = _load_model()
		
		# Convert PIL Image to format expected by keras preprocessing (same as predict1.py)
		# Resize and convert PIL image first
		pil_image_rgb = pil_image.convert('RGB').resize((224, 224))
		
		# Use keras_image.img_to_array to match predict1.py exactly
		img_array = keras_image.img_to_array(pil_image_rgb) / 255.0
		img_array = np.expand_dims(img_array, axis=0)
		
		# 5. Predict (same as predict1.py)
		predictions = model.predict(img_array, verbose=0)
		
		# Convert to same format as backend expects
		results: list[dict[str, float]] = []
		for i, prob in enumerate(predictions[0]):
			label = class_names.get(i, f"Class_{i}")
			results.append({
				"label": label,
				"probability": float(prob)
			})
		
		# Sort by probability (highest first)
		results.sort(key=lambda x: x["probability"], reverse=True)
		
		return results
	except Exception as e:
		# Add error info for debugging
		import traceback
		error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
		raise RuntimeError(error_msg) from e


