
# 🚢 Illegal Fishing Hotspot Detection Using Machine Learning

### **Mining AIS Big Data to Identify Hidden Fishing Activities in the Mediterranean Sea**

---

## 📌 **Project Overview**

Illegal, unreported, and unregulated (IUU) fishing threatens marine ecosystems and global food security.
This project uses **Machine Learning + AIS (Automatic Identification System) vessel data** to detect hidden fishing patterns and identify possible **illegal fishing hotspots** in the Mediterranean Sea.

We process AIS data, detect suspicious behavior (like identity gaps), cluster dangerous zones, and visualize hotspots on an interactive map using **Streamlit**, **Folium**, and **Heatmaps**.

A KNN-based prediction system is also included to check **whether a new vessel belongs to any hotspot**.

---

## 🧠 **Key Features**

### ✔️ **AIS Data Preprocessing**

* Sorting by MMSI and timestamp
* Calculating gaps between consecutive signals
* Detecting “identity hiding gaps” (AIS turned off)

### ✔️ **Suspicious Vessel Detection**

* Identifying vessels with abnormal behavior
* Filtering real fishing-related points

### ✔️ **Hotspot Clustering (DBSCAN)**

* Density-based clustering to detect illegal-fishing zones
* Automatic centroid extraction
* Cluster quality metrics (Silhouette Score, Davies–Bouldin Score)

### ✔️ **KNN Prediction Model**

Predicts the cluster of a new vessel based on:

* Latitude
* Longitude
* Speed
* MMSI

If the vessel is too far from known clusters → returns **Cluster = -1 (Safe / Not suspicious)**

### ✔️ **Interactive Streamlit Web App**

* ⚡ Heatmap visualization
* 📍 Cluster centroid markers
* 📄 Raw data display
* 🤖 Live vessel cluster prediction
* 🖤 Black theme with custom UI
* 📷 Optional Mediterranean Sea background image

---

## 🗂️ **Tech Stack**

### **Machine Learning**

* Python
* Scikit-Learn
* DBSCAN
* K-Nearest Neighbors (KNN)
* Clustering metrics

### **Data**

* AIS Big Data (simulated dataset of ~30,000 rows)

### **Backend**

* Python
* Pandas
* NumPy

### **Frontend / Visualization**

* Streamlit
* Folium
* HeatMap Plugin
* Custom HTML/CSS styling

---

## 📌 **Architecture / Pipeline**

```
AIS Big Data → Clean & Sort Data →
Detect AIS Gaps → Extract Suspicious Points →
Cluster Using DBSCAN → Generate Hotspot Centroids →
Train KNN Model → Deploy Streamlit Dashboard →
Predict Vessel Hotspot Risk
```

---

## 🚀 **How To Run This Project**

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/illegal-fishing-hotspots.git
cd illegal-fishing-hotspots
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

### 4️⃣ Open in your browser

Usually at:

```
http://localhost:8501
```

---

## 🔮 **Future Improvements**

* Integrate real-time AIS streaming
* Use satellite imagery for validation
* Implement LSTM-based vessel behavior forecasting
* Add anomaly detection (Isolation Forest)
* Build a full maritime risk dashboard

---

## 👨‍💻 Author

**Kevin Manjila**
Machine Learning Developer
[LinkedIn Profile](www.linkedin.com/in/kevin-manjila)

---



