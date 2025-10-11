import { doc, setDoc } from "firebase/firestore"; 


async function storeDriverData(db) {
  const drivers = {
    "driver_1": { "safetyScore": 87, "lastTrip": { "speedingIncidents": 2, "hardBraking": 4, "location": "29.275008, 73.972625" } },
    "driver_2": { "safetyScore": 75, "lastTrip": { "speedingIncidents": 1, "hardBraking": 2, "location": "21.113586, 73.569834" } },
    "driver_3": { "safetyScore": 95, "lastTrip": { "speedingIncidents": 0, "hardBraking": 1, "location": "26.274134, 94.065021" } },
    "driver_4": { "safetyScore": 61, "lastTrip": { "speedingIncidents": 3, "hardBraking": 2, "location": "13.308159, 76.271268" } },
    "driver_5": { "safetyScore": 68, "lastTrip": { "speedingIncidents": 2, "hardBraking": 5, "location": "20.765151, 92.732738" } },
    
  };
}
  for (const [id, data] of Object.entries(drivers)) {
    await setDoc(doc(db, "drivers", id), data);
  }



import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";

const firebaseConfig = {
 apiKey: "YOUR_API_KEY",
 authDomain: "YOUR_AUTH_DOMAIN",
 projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
import { doc, setDoc } from "firebase/firestore"; 

await setDoc(doc(db, "drivers", "driver_123"), {
  safetyScore: 85,
  lastTrip: {
    speedingIncidents: 2,
    hardBraking: 1,
    location: "37.7749, -122.4194"
  }
});



import { getDoc } from "firebase/firestore";

const driverRef = doc(db, "drivers", "driver_123");
const driverSnap = await getDoc(driverRef);

if (driverSnap.exists()) {
  document.getElementById("safety-score").textContent = driverSnap.data().safetyScore;
}


