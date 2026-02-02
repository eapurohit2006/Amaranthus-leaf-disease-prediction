const API_BASE = localStorage.getItem("API_BASE") || "http://localhost:8000";

function setNav() {
	const nav = document.getElementById("nav");
	if (!nav) return;
	const token = localStorage.getItem("token");
	const path = location.pathname.split("/").pop() || "index.html";
	nav.innerHTML = `
		<a href="index.html" class="${path === 'index.html' ? 'active' : ''}">Home</a>
		<a href="about.html" class="${path === 'about.html' ? 'active' : ''}">About</a>
		<a href="predict.html" class="${path === 'predict.html' ? 'active' : ''}">Predict</a>
		<span class="spacer"></span>
		${token ? '<a href="#" id="logout">Logout</a>' : '<a href="login.html" class="' + (path==='login.html'?'active':'') + '">Login</a><a href="signup.html" class="' + (path==='signup.html'?'active':'') + '">Sign Up</a>'}
	`;
	const logout = document.getElementById("logout");
	if (logout) {
		logout.addEventListener("click", (e) => {
			e.preventDefault();
			localStorage.removeItem("token");
			location.href = "index.html";
		});
	}
}

async function api(path, opts = {}) {
	const token = localStorage.getItem("token");
	const headers = new Headers(opts.headers || {});
	if (token) headers.set("Authorization", `Bearer ${token}`);
	return fetch(`${API_BASE}${path}`, { ...opts, headers });
}

document.addEventListener("DOMContentLoaded", setNav);






