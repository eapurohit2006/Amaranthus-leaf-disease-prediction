import json

with open("class_indices_resnet50v2.json", "r") as f:
    data = json.load(f)

print(f"Keys: {[repr(k) for k in data.keys()]}")
