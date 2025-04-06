// Sample Data
let userPoints = 0;
let userCrossings = 0;
let userBadges = 0;

const leaderboardData = [
  { rank: 1, user: "User1", points: 150 },
  { rank: 2, user: "User2", points: 120 },
  { rank: 3, user: "User3", points: 100 },
];

const notifications = [
  "You earned 10 points for using FOB!",
  "New badge unlocked: FOB Beginner!",
  "Reminder: Use FOB to earn rewards.",
];

// Update User Stats
function updateStats() {
  document.getElementById("points").textContent = userPoints;
  document.getElementById("crossings").textContent = userCrossings;
  document.getElementById("badges").textContent = userBadges;
}

// Populate Leaderboard
function populateLeaderboard() {
  const leaderboardTable = document.getElementById("leaderboard-table").getElementsByTagName("tbody")[0];
  leaderboardData.forEach((entry) => {
    const row = leaderboardTable.insertRow();
    row.insertCell(0).textContent = entry.rank;
    row.insertCell(1).textContent = entry.user;
    row.insertCell(2).textContent = entry.points;
  });
}

// Populate Notifications
function populateNotifications() {
  const notificationList = document.getElementById("notification-list");
  notifications.forEach((notification) => {
    const li = document.createElement("li");
    li.textContent = notification;
    notificationList.appendChild(li);
  });
}

// Simulate QR Code Scan
document.getElementById("scan-btn").addEventListener("click", () => {
  userPoints += 10;
  userCrossings += 1;
  if (userCrossings === 1) {
    userBadges += 1;
    notifications.push("New badge unlocked: FOB Beginner!");
    populateNotifications();
  }
  updateStats();
  document.getElementById("scan-result").textContent = "QR Code Scanned! 10 points earned.";
});

// Initialize Dashboard
updateStats();
populateLeaderboard();
populateNotifications();