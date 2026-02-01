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
      /* Add App Name to top-left header */
    .stAppHeader::before {
        content: "MediFishSpot AI";
        position: absolute;
        left: 80px;  /* adjust based on sidebar button */
        top: 10px;
        font-size: 24px;
        font-weight: bold;
        color: white; /* black color */
        z-index: 9999;
    }
    .stAppHeader{
            background-color:black;
            }

    /* Global Light Theme */
    .stApp {
        background-color: #f4f4f9 !important;
        font-family: 'Segoe UI', sans-serif;
        color: #222 !important;
    }

    h1, h2, h3 {
        color: #0a66c2 !important;
        animation: fadeInDown 1s ease;
    }

    /* Subheader color */
    .stMarkdown h3, h3 {
        color: #333 !important;
    }


    /* -----------------------
       ✨ Animated Glass Cards
       ----------------------- */
    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #d0d0d0;
        box-shadow: 0px 4px 20px rgba(150,150,150,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.35s ease, box-shadow 0.35s ease;
        animation: fadeIn 1s ease;
    }

    .glass-card:hover {
        transform: translateY(-6px);
        box-shadow: 0px 10px 30px rgba(100,100,100,0.3);
    }

    /* Fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }


    /* -----------------------
       ✨ Sidebar Styling
       ----------------------- */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
        animation: fadeIn 1s ease;
    }

    [data-testid="stSidebar"] * {
        color: #333 !important;
        font-size: 16px !important;
    }

    /* Radio button options spacing */
    div[role="radiogroup"] > label {
        padding: 10px 0;
        margin-bottom: 6px;
        transition: 0.3s;
        
    }
    div[role="radiogroup"] > label:hover {
        transform: translateX(6px);
        color: #0a66c2 !important;
        font-weight: bold;
    }


    /* -----------------------
       ✨ Buttons Animated
       ----------------------- */
    .stButton>button {
        background-color: #0a66c2 !important;
        color: white !important;
        padding: 10px 18px;
        border-radius: 8px;
        border: none;
        transition: 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #004182 !important;
        transform: scale(1.03);
        box-shadow: 0px 4px 10px rgba(0,0,0,0.25);
    }


    /* -----------------------
       ✨ Input Box Styling
       ----------------------- */
    input, textarea {
        border-radius: 6px !important;
        border: 1px solid #aaaaaa !important;
    }

    input:focus, textarea:focus {
        border: 1px solid #0a66c2 !important;
        box-shadow: 0px 0px 6px rgba(10,102,194,0.3);
    }
    
            
        /* Streamlit Alerts Text Color */
    .stAlert p, 
    .stAlert div[data-testid="stAlertContentWarning"] p, 
    .stAlert div[data-testid="stAlertContentError"] p, 
    .stAlert div[data-testid="stAlertContentSuccess"] p {
        color: #000000 !important;  /* black text */
    }

    /* Optional: make warning background light yellow */
    .stAlert[data-testid="stAlert"] {
        background-color: #fffacd !important;  /* light yellow */
    }

    /* Optional: make error background light pink */
    .stAlert[data-testid="stAlert"][role="alert"][data-baseweb="notification"][class*="stAlertContainer"] {
        background-color: #ffe4e1 !important;  /* light pink */
    }

     /* FIX: Make number_input labels visible again */
    div[data-testid="stNumberInput"] label p {
        color: black !important;   /* Blue label text */
        font-weight: 500 !important;
        font-size: 18px !important;
    }

    /* Alternatively black text:
    div[data-testid="stNumberInput"] label p {
        color: black !important;
    }
    */

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
