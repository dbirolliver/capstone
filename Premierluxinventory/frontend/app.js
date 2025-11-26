let currentUserRole = 'admin'; // Change to 'staff' to test

function showPage(page) {
  // Hide all sections
  document.getElementById('dashboard-section').style.display = 'none';
  document.getElementById('inventory-section').style.display = 'none';
  document.getElementById('branches-section').style.display = 'none';
  document.getElementById('staff-section').style.display = 'none';

  // Show selected section
  document.getElementById(page + '-section').style.display = 'block';

  // Update page title
  document.getElementById('page-title').textContent =
    page.charAt(0).toUpperCase() + page.slice(1);

  // Show/hide admin-only cards/features
  if (currentUserRole === 'admin') {
    document.getElementById('admin-section').style.display = 'block';
    document.getElementById('role-indicator').textContent = 'Role: Admin';
  } else {
    document.getElementById('admin-section').style.display = 'none';
    document.getElementById('role-indicator').textContent = 'Role: Staff';
  }

  // On mobile, close sidebar after navigation
  if (window.innerWidth <= 900) {
    document.getElementById('sidebar').classList.remove('open');
  }
}

// Sidebar toggle for mobile
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}

// Show dashboard by default
showPage('dashboard');

