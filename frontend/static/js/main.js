document.addEventListener("DOMContentLoaded", () => {
  const themeToggle = document.getElementById("themeToggle");
  const menuBtn = document.getElementById("menuBtn");
  const navLinks = document.getElementById("navLinks");

  const savedTheme = localStorage.getItem("foodlens_theme") || localStorage.getItem("freshlens_theme") || localStorage.getItem("safebite_theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcon(savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const nextTheme = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      localStorage.setItem("foodlens_theme", nextTheme);
      updateThemeIcon(nextTheme);
    });
  }

  function updateThemeIcon(theme) {
    if (themeToggle) {
      themeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
      themeToggle.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
    }
  }

  // 2. Mobile Drawer Navigation
  if (menuBtn && navLinks) {
    menuBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      navLinks.classList.toggle("active");
      const isOpen = navLinks.classList.contains("active");
      menuBtn.textContent = isOpen ? "✕" : "☰";
    });

    // Close menu when clicking outside
    document.addEventListener("click", (e) => {
      if (!navLinks.contains(e.target) && e.target !== menuBtn) {
        navLinks.classList.remove("active");
        menuBtn.textContent = "☰";
      }
    });
  }

  // 3. Register PWA Service Worker
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker
        .register("/static/js/sw.js")
        .then((reg) => {
          console.log("FoodLens-AI ServiceWorker registered:", reg.scope);
        })
        .catch((err) => {
          console.warn("ServiceWorker registration skipped:", err);
        });
    });
  }
});
