// Initialize the Risk Map
const map = L.map('risk-map').setView([37.7749, -122.4194], 10); // San Francisco
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Add risk markers
const riskData = [
  { lat: 37.7749, lon: -122.4194, risk: 0.8 }, // High risk
  { lat: 34.0522, lon: -118.2437, risk: 0.4 }, // Medium risk
];

riskData.forEach(({ lat, lon, risk }) => {
  const color = risk < 0.3 ? 'green' : risk < 0.7 ? 'yellow' : 'red';
  L.circleMarker([lat, lon], {
    radius: 5 * (1 + risk),
    color: color,
    fill: true,
    fillColor: color
  }).bindPopup(`Risk: ${(risk * 100).toFixed(2)}%`).addTo(map);
});

// Telematics and Road Safety Score
const telematicsData = {
  speedingIncidents: 2,
  hardBraking: 1,
  laneChangeViolations: 0
};

function calculateRoadSafetyScore(data) {
  const score = 100 - (
    data.speedingIncidents * 10 +
    data.hardBraking * 5 +
    data.laneChangeViolations * 3
  );
  return Math.max(0, score);
}

function offerIncentives(score) {
  if (score >= 90) return "10% insurance discount, 100 reward points";
  if (score >= 70) return "5% insurance discount, 50 reward points";
  return "No incentives. Improve your driving habits.";
}

const safetyScore = calculateRoadSafetyScore(telematicsData);
const driverRef = firebase.firestore().collection("drivers").doc("driver_123");
driverRef.get().then((doc) => {
  if (doc.exists) {
    document.getElementById('safetyScore').textContent = doc.data().safetyScore;
  }
});
document.getElementById('incentives').textContent = `Incentives: ${offerIncentives(safetyScore)}`;

// Driver Distraction Detection
function detectDriverDistractions() {
  const distractions = {
    phoneUsage: Math.random() > 0.5,
    drowsiness: Math.random() > 0.5,
    distractedGlances: Math.random() > 0.5
  };
  return distractions;
}

function alertDriver(distractions) {
  const alerts = [];
  if (distractions.phoneUsage) alerts.push("Warning: Phone usage detected!");
  if (distractions.drowsiness) alerts.push("Warning: Drowsiness detected!");
  if (distractions.distractedGlances) alerts.push("Warning: Distracted glances detected!");
  return alerts;
}

const distractions = detectDriverDistractions();
const alerts = alertDriver(distractions);
document.getElementById('distraction-alerts').innerHTML = alerts.join('<br>');

// Citizen Reporting System
const hazardReports = [];

document.getElementById('hazard-report-form').addEventListener('submit', (e) => {
  e.preventDefault();

  const hazardType = document.getElementById('hazard-type').value;
  const hazardPhoto = document.getElementById('hazard-photo').files[0];
  const location = { lat: map.getCenter().lat, lon: map.getCenter().lng };

  const reader = new FileReader();
  reader.onload = () => {
    const report = {
      hazardType,
      location,
      photoUrl: reader.result,
      timestamp: new Date().toLocaleString()
    };
    hazardReports.push(report);
    displayReportedHazards();
  };
  reader.readAsDataURL(hazardPhoto);
});

function displayReportedHazards() {
  const reportedHazardsDiv = document.getElementById('reported-hazards');
  reportedHazardsDiv.innerHTML = hazardReports.map((report, index) => `
    <div class="hazard-card">
      <h3>Report #${index + 1}: ${report.hazardType}</h3>
      <p>Location: ${report.location.lat.toFixed(4)}, ${report.location.lon.toFixed(4)}</p>
      <p>Reported at: ${report.timestamp}</p>
      <img src="${report.photoUrl}" alt="Hazard Photo" />
    </div>
  `).join('');
}


