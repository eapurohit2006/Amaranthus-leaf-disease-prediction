import sys
from pathlib import Path
from PIL import Image

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    from backend.tf_inference import load_tf_predict
    print("Import successful")
    
    predict_fn = load_tf_predict()
    print("Model loaded")
    
    img_path = "resnet50v2_training.png"
    if Path(img_path).exists():
        img = Image.open(img_path)
        results = predict_fn(img)
        print("Predictions:")
        for r in results:
            print(f"  {r['label']}: {r['probability']:.4f}")
    else:
        print("Image not found")
        
except Exception as e:
    import traceback
    traceback.print_exc()
