const form = document.getElementById("signup-form");
form.addEventListener("submit", async (e) => {
	e.preventDefault();
	const full_name = document.getElementById("full_name").value.trim();
	const email = document.getElementById("email").value.trim();
	const password = document.getElementById("password").value;
	const res = await api("/auth/signup", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ full_name, email, password })
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		alert(err.detail || "Signup failed");
		return;
	}
	const data = await res.json();
	localStorage.setItem("token", data.access_token);
	location.href = "predict.html";
});






