const form = document.getElementById("predict-form");
const input = document.getElementById("image");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("preview-img");
const results = document.getElementById("results");
const list = document.getElementById("predictions");

input.addEventListener("change", () => {
	const file = input.files?.[0];
	if (!file) {
		preview.classList.add("hidden");
		previewImg.src = "";
		return;
	}
	previewImg.src = URL.createObjectURL(file);
	preview.classList.remove("hidden");
});

form.addEventListener("submit", async (e) => {
	e.preventDefault();
	const file = input.files?.[0];
	if (!file) return;
	const fd = new FormData();
	fd.append("image", file);
	list.innerHTML = "";
	results.classList.add("hidden");
	const res = await api("/predict", { method: "POST", body: fd });
	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		alert(err.detail || "Prediction failed");
		return;
	}
	const data = await res.json();
	render(data.predictions || []);
});

function render(items) {
	list.innerHTML = "";
	if (!items.length) {
		list.innerHTML = `<li>No predictions.</li>`;
		results.classList.remove("hidden");
		return;
	}
	items.forEach((p) => {
		const li = document.createElement("li");
		li.innerHTML = `
			<div class="top"><span>${p.label}</span><span class="prob">${(p.probability*100).toFixed(1)}%</span></div>
			<ul class="precautions">${(p.precautions||[]).map(x=>`<li>${x}</li>`).join("")}</ul>
		`;
		list.appendChild(li);
	});
	results.classList.remove("hidden");
}



