import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Illegal Fishing Hotspots",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme styling
st.markdown("""
    <style>
    body {
        background-color: #0d1117;
        color: #e6edf3;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    .stApp {
        background-color: #0d1117;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("⛴️ Mediterranean Illegal Fishing Hotspots")
st.markdown("### Detected using Machine Learning & AIS Big Data")
st.write("---")

# Load data
hotspots = pd.read_csv("hotspot_points.csv")
centroids = pd.read_csv("cluster_centroids.csv")

# Create folium map
m = folium.Map(location=[38, 15], zoom_start=4, tiles="CartoDB dark_matter")

# Add heatmap
HeatMap(hotspots[['lat','lon']].values.tolist(),
        radius=15,
        blur=10).add_to(m)

# Add centroid markers
for _, row in centroids.iterrows():
    folium.Marker(
        [row['center_lat'], row['center_lon']],
        popup=f"Cluster {row['cluster']} | {row['count']} events",
        icon=folium.Icon(color="red")
    ).add_to(m)

# Show map inside Streamlit
st_folium(m, width=1400, height=650)
