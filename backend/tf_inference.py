from __future__ import annotations
from typing import Callable
from pathlib import Path
import json
import numpy as np
from PIL import Image


def load_tf_predict() -> Callable[[Image.Image], list[dict[str, float]]]:
	"""
	Loads a TensorFlow/Keras model and returns a predict(image) callable.
	It auto-detects model filename and uses class_indices_resnet50v2.json.
	"""
	try:
		import tensorflow as tf  # noqa: F401
		from tensorflow.keras.preprocessing import image as keras_image
	except Exception as e:
		raise RuntimeError("TensorFlow is required for TF model inference but is not installed.") from e

	model_path = None
	for candidate in [
		"amaranthus_resnet50v2_high_acc.keras",
		"amaranthus_resnet50v2.keras",
	]:
		if Path(candidate).exists():
			model_path = candidate
			break
	if model_path is None:
		raise FileNotFoundError("Keras model file not found (expected amaranthus_resnet50v2_high_acc.keras or amaranthus_resnet50v2.keras)")

	cls_json = Path("class_indices_resnet50v2.json")
	if not cls_json.exists():
		raise FileNotFoundError("class_indices_resnet50v2.json not found")
	with cls_json.open("r", encoding="utf-8") as f:
		class_indices = json.load(f)
	# Reverse mapping: {index: name}
	index_to_name = {v: k for k, v in class_indices.items()}

	# Lazily load model
	import tensorflow as tf
	model = tf.keras.models.load_model(model_path)

	def predict(pil_image: Image.Image) -> list[dict[str, float]]:
		img = pil_image.resize((224, 224))
		arr = np.asarray(img, dtype=np.float32) / 255.0
		arr = np.expand_dims(arr, axis=0)
		probs = model.predict(arr)[0]
		results: list[dict[str, float]] = []
		for idx, p in enumerate(probs):
			label = index_to_name.get(idx, str(idx))
			results.append({"label": label, "probability": float(p)})
		# Sort high to low
		results.sort(key=lambda x: x["probability"], reverse=True)
		return results

	return predict
