import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

import pickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load KNN cluster predictor
with open("knn_cluster_model.pkl", "rb") as f:
    knn = pickle.load(f)

def predict_cluster(lat, lon, threshold=0.5):
    """
    Predict cluster for new point.
    If too far from any cluster → return -1
    """

    # Predict using KNN
    predicted = knn.predict([[lat, lon]])[0]

    # Distance to nearest cluster point
    dist, idx = knn.kneighbors([[lat, lon]], n_neighbors=1)
    nearest_distance = dist[0][0]

    # If too far, treat as noise/outlier
    if nearest_distance > threshold:
        return -1

    return predicted



# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Illegal Fishing Hotspots",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM DARK THEME CSS + PREMIUM UI DESIGN
# ------------------------------------------------------------
st.markdown("""
    <style>

    /* Global Dark Theme */
    .stApp {
        background-color: #0d1117 !important;
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Page Titles */
    h1, h2, h3 {
        color: #58a6ff !important;
    }

    /* 🔥 Force ALL Subheaders to white */
    .stMarkdown h3, 
    h3, 
    .css-10trblm, 
    .css-1v3fvcr, 
    .css-ztfqz8, 
    .css-16idsys {
        color: #ffffff !important;
    }

    /* Paragraphs & labels */
    p, label, span {
        color: #ffffff !important;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(22, 27, 34, 0.55);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.35);
        backdrop-filter: blur(8px);
        transition: 0.3s;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0px 8px 25px rgba(0,0,0,0.55);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    /* Sidebar text forced white */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Radio button text white */
    .css-qrbaxs, .css-1pcexqc, .css-16huue1 {
        color: white !important;
    }

    /* Increase spacing between sidebar radio options */
    div[role="radiogroup"] > label {
        padding-top: 10px;
        padding-bottom: 10px;
        margin-bottom: 6px;
    }

    /* Info box text */
    .stAlert > div {
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)



# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title' style='color:white;font-size:22px;font-weight:bold;'>Navigation</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select a View",
    ["Heatmap", "Cluster Centroids", "Raw Data", "Predict Cluster"],
    index=0
)

st.sidebar.write("---")


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.markdown("<h1>Illegal Fishing Hotspots in the Mediterranean</h1>", unsafe_allow_html=True)
st.markdown("<h3>Detected using Machine Learning & AIS Big Data</h3>", unsafe_allow_html=True)
st.write("")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
hotspots = pd.read_csv("hotspot_points.csv")
centroids = pd.read_csv("cluster_centroids.csv")

# ------------------------------------------------------------
# PAGE: HEATMAP
# ------------------------------------------------------------
if page == "Heatmap":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Heatmap of Suspicious Fishing Activity")

    # Create map
    m = folium.Map(location=[38, 15], zoom_start=4, tiles="CartoDB dark_matter")
    HeatMap(hotspots[['lat','lon']].values.tolist(),
            radius=15, blur=10).add_to(m)

    st_map = st_folium(m, width=1400, height=650)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# PAGE: CLUSTER CENTROIDS
# ------------------------------------------------------------
elif page == "Cluster Centroids":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Hotspot Cluster Centers")

    m = folium.Map(location=[38, 15], zoom_start=4, tiles="CartoDB dark_matter")

    # Add centroid markers
    for _, row in centroids.iterrows():
        folium.Marker(
            [row['center_lat'], row['center_lon']],
            popup=f"Cluster {row['cluster']} | {row['count']} events",
            icon=folium.Icon(color="red")
        ).add_to(m)

    st_map = st_folium(m, width=1400, height=650)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# PAGE: RAW DATA
# ------------------------------------------------------------
elif page == "Raw Data":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Raw Hotspot Data")

    st.write("### Hotspot Events")
    st.dataframe(hotspots, height=500)

    st.write("### Cluster Summary")
    st.dataframe(centroids)

    st.markdown("</div>", unsafe_allow_html=True)





# ------------------------------------------------------------
# PAGE: PREDICT CLUSTER
# ------------------------------------------------------------
elif page == "Predict Cluster":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Predict Vessel Cluster")

    st.write("Enter vessel details to predict whether it belongs to any illegal fishing hotspot cluster.")

    mmsi = st.number_input("Enter MMSI", min_value=100000000, max_value=999999999)
    lat = st.number_input("Enter Latitude", format="%.6f")
    lon = st.number_input("Enter Longitude", format="%.6f")
    speed = st.number_input("Enter Speed (knots)", format="%.2f")

    if st.button("Predict Cluster"):

        if lat == 0 and lon == 0:
            st.error("Please enter valid coordinates.")
        else:
            cluster = predict_cluster(lat, lon)

            if cluster == -1:
                st.warning("This vessel is NOT near any known illegal fishing hotspot (Cluster = -1).")
            else:
                st.success(f"This vessel belongs to Cluster {cluster}.")

    st.markdown("</div>", unsafe_allow_html=True)
