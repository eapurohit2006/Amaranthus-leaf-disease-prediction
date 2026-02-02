from typing import Callable, Any
from importlib import util
from pathlib import Path
from PIL import Image


class ModelNotFound(Exception):
	pass


def load_external_predict() -> Callable[[Image.Image], list[dict[str, float]]]:
	"""
	Looks for model/inference.py with a function predict(image: PIL.Image.Image) -> list[{'label','probability'}]
	"""
	module_path = Path("model") / "inference.py"
	if not module_path.exists():
		raise ModelNotFound(f"model/inference.py not found at {module_path.absolute()}")

	spec = util.spec_from_file_location("external_inference", str(module_path))
	if spec is None or spec.loader is None:
		raise ModelNotFound(f"Unable to create spec for model/inference.py at {module_path.absolute()}")
	
	try:
		mod = util.module_from_spec(spec)
		spec.loader.exec_module(mod)  # type: ignore[attr-defined]
	except Exception as e:
		raise ModelNotFound(f"Error executing model/inference.py: {e}") from e

	predict_fn: Any = getattr(mod, "predict", None)
	if not callable(predict_fn):
		raise ModelNotFound("model/inference.py must define a callable predict(image) function.")
	return predict_fn  # type: ignore[return-value]


def load_predict_function() -> Callable[[Image.Image], list[dict[str, float]]]:
	"""
	Try in order:
	1) model/inference.py::predict
	2) Local TensorFlow Keras files via backend.tf_inference
	"""
	# 1) external inference
	try:
		return load_external_predict()
	except Exception as e:
		# Log error but continue to fallback
		print(f"Failed to load model/inference.py: {e}")
		pass
	# 2) TF inference fallback
	try:
		from .tf_inference import load_tf_predict
		return load_tf_predict()
	except Exception as e:
		raise ModelNotFound(str(e)) from e

