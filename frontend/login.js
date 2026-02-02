const form = document.getElementById("login-form");
form.addEventListener("submit", async (e) => {
	e.preventDefault();
	const email = document.getElementById("email").value.trim();
	const password = document.getElementById("password").value;
	const res = await api("/auth/login", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ email, password })
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		alert(err.detail || "Login failed");
		return;
	}
	const data = await res.json();
	localStorage.setItem("token", data.access_token);
	location.href = "predict.html";
});






