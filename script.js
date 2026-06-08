// --- NAVBAR ACTIVE STATE LOGIC ---
document.addEventListener("DOMContentLoaded", () => {
  const currentPage = window.location.pathname.split("/").pop();
  document.querySelectorAll(".nav-links a").forEach(link => {
    if (link.getAttribute("href") === currentPage) {
      link.classList.add("active");
    }
  });

  // Check auth state on load for the account page
  if (currentPage === 'account.html' || currentPage === '') {
    checkAuthState();
  }
});

// --- DETECTOR PAGE LOGIC ---
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.input-group').forEach(group => group.classList.remove('active'));
  event.target.classList.add('active');
  setTimeout(() => { document.getElementById('input-' + tabId).classList.add('active'); }, 50);
}

// Dummy AI Detector Run
function runDetector() {
  const btn = document.querySelector('.cyber-btn');
  btn.innerText = "ANALYZING NEURAL DATA...";
  setTimeout(() => {
    alert("Verification Complete! (Connect Python Backend to see detailed results)");
    btn.innerText = "INITIATE SCAN";
  }, 2000);
}

// --- ACCOUNT PAGE LOGIC ---
function handleLogin(event) {
  event.preventDefault();
  // Dummy login - save to local storage
  localStorage.setItem("verifiai_auth", "true");
  checkAuthState();
}

function handleLogout() {
  localStorage.removeItem("verifiai_auth");
  checkAuthState();
}

function checkAuthState() {
  const loginView = document.getElementById("login-view");
  const dashboardView = document.getElementById("dashboard-view");

  if (!loginView || !dashboardView) return; // Prevent errors on other pages

  if (localStorage.getItem("verifiai_auth") === "true") {
    loginView.style.display = "none";
    dashboardView.style.display = "block";
  } else {
    loginView.style.display = "block";
    dashboardView.style.display = "none";
  }
}
// --- TEAM MEMBER MODAL LOGIC ---
function openModal(name, role, imgSrc) {
  const modal = document.getElementById('member-modal');
  document.getElementById('modal-img').src = imgSrc;
  document.getElementById('modal-name').innerText = name;
  document.getElementById('modal-role').innerText = role;

  modal.classList.add('active');
}

function closeModal() {
  document.getElementById('member-modal').classList.remove('active');
}

// Close modal if user clicks outside the box
window.onclick = function (event) {
  const modal = document.getElementById('member-modal');
  if (event.target === modal) {
    closeModal();
  }
}

// --- ACCOUNT PAGE INSTAGRAM TABS ---
function switchProfileTab(tabName) {
  // Hide all tab content
  document.querySelectorAll('.profile-tab-content').forEach(content => {
    content.style.display = 'none';
  });
  // Remove active class from buttons
  document.querySelectorAll('.profile-tab').forEach(btn => {
    btn.classList.remove('active');
  });

  // Show selected content and activate button
  document.getElementById('tab-' + tabName).style.display = 'grid';
  event.target.classList.add('active');
}