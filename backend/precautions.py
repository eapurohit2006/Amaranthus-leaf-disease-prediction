DISEASE_PRECAUTIONS: dict[str, list[str]] = {
	"Healthy": [
		"Maintain regular watering and avoid leaf wetness at night.",
		"Inspect weekly to catch early signs of disease.",
		"Ensure good airflow; avoid overcrowding.",
	],
	"Leaf Blight": [
		"Remove and destroy infected leaves to reduce spread.",
		"Apply copper-based fungicide as per label guidance.",
		"Rotate crops and avoid overhead irrigation.",
	],
	"Powdery Mildew": [
		"Improve air circulation and reduce humidity.",
		"Use sulfur or potassium bicarbonate sprays early.",
		"Avoid excess nitrogen fertilization.",
	],
	"Leaf Spot": [
		"Prune affected foliage and disinfect tools.",
		"Mulch to limit soil splash and spore spread.",
		"Consider appropriate fungicide if severe.",
	],
}


def get_precautions(label: str) -> list[str]:
	return DISEASE_PRECAUTIONS.get(label, [
		"Isolate affected plants and monitor closely.",
		"Consult local extension service for targeted advice.",
	])


