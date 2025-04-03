# Contains the UI logic 

import streamlit as st
import folium
import osmnx as ox
import networkx as nx
import leafmap.foliumap as leafmap
import sys
import os
import json
import branca.colormap as cm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.solver import solve, parse_votes_str  

BASEMAPS = ['Satellite', 'Roadmap', 'Terrain', 'Hybrid', 'OpenStreetMap']

def clear_coordinates():
    st.session_state["start_lat"] = ""
    st.session_state["start_lon"] = ""
    st.session_state["end_lat"] = ""
    st.session_state["end_lon"] = ""

st.set_page_config(page_title="🗺️ Military Route Planner", layout="wide")

# ====== SIDEBAR LOGIC - includes starting example from backend/testing.py main function ======
with st.sidebar:
    st.title("Route Planning Settings")
    st.markdown("Military route planning assistant that finds optimal paths based on commander's intent.")
    basemap = st.selectbox("Choose basemap", BASEMAPS)
    if basemap in BASEMAPS[:-1]:
        basemap = basemap.upper()
    st.subheader("Commander's Intent")
    commander_intent = st.text_area(
        "Tactical preferences:",
        "I am interested in terrain where there are no enemy operatives nearby, "
        "where the road is unobstructed. I do not wish to pass by enemy radar, "
        "enemy platoons or bases. I do not wish to cross any water."
    )
    st.subheader("Coordinates")
    start_lat = st.number_input("Start Latitude", value=47.108750, format="%.7f", key="start_lat")
    start_lon = st.number_input("Start Longitude", value=37.523804, format="%.7f", key="start_lon")
    end_lat = st.number_input("End Latitude", value=47.121474, format="%.7f", key="end_lat")
    end_lon = st.number_input("End Longitude", value=37.542343, format="%.7f", key="end_lon")
    st.button("Clear coordinates", on_click=clear_coordinates)
    st.info("This tool helps military commanders plan routes that account for tactical considerations and threats in the area of operations.")

# ====== MAIN PAGE LOGIC ======
st.title("Military Route Planning Assistant")

# Initialize map centered on start coordinates
m = leafmap.Map(center=(start_lat, start_lon), zoom=16)
m.add_basemap(basemap)

if all([start_lat, start_lon, end_lat, end_lon]):
    try:
        # Get routes and additional info from solve function
        result = solve(
            center_coord=(start_lat, start_lon),
            start_coord=(start_lat, start_lon),
            dst_coord=(end_lat, end_lon),
            commander_intent=commander_intent
        )

        # Create graph (for markers, etc.)
        G = ox.graph_from_point((start_lat, start_lon), dist=1000, network_type='all')
        
        # Add markers for start and end points
        m.add_marker(location=[start_lat, start_lon],
                     popup="Start Point",
                     icon=folium.Icon(color='red', icon='play', prefix='fa'))
        m.add_marker(location=[end_lat, end_lon],
                     popup="End Point",
                     icon=folium.Icon(color='green', icon='stop', prefix='fa'))
        
        # Plot both paths on the map
        naive_path = result["naive_path"]
        informed_path = result["informed_path"]

        folium.PolyLine(naive_path, color='blue', weight=3, opacity=0.8, popup='Shortest Path').add_to(m)
        folium.PolyLine(informed_path, color='red', weight=3, opacity=0.8, popup='Tactical Path').add_to(m)
        
        votes_dict = {}
        for bbox_info in result["bbox_info"]:
            bbox = tuple(bbox_info["bounds"])
            votes_dict[bbox] = bbox_info["weight"]

        # Determine min and max vote values for the color scale
        vote_values = list(votes_dict.values())
        min_vote = min(vote_values)
        max_vote = max(vote_values)
        
        # Create a reversed colormap 
        colormap = cm.LinearColormap(
            colors=cm.linear.RdYlGn_09.colors[::-1],
            index=cm.linear.RdYlGn_09.index[::-1],
            vmin=min_vote,
            vmax=max_vote,
        )
        
        # Iterate over bounding boxes to add a shaded polygon overlay
        for bbox_info in result["bbox_info"]:
            bbox = bbox_info["bounds"]  # tuple: (north, south, east, west)
            north, south, east, west = bbox
            # Retrieve the vote value; if not found, default to max_vote (safer)
            vote = votes_dict.get(tuple(bbox), max_vote)
            fill_color = colormap(vote)
            # Build polygon coordinates for folium: [(lat, lon), ...]
            poly_coords = [(north, west), (north, east), (south, east), (south, west)]
            # Add polygon to the map with tooltip displaying vote and summary
            folium.Polygon(
                locations=poly_coords,
                color='gray',
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.4,
                weight=1,
                tooltip=f"Description: {bbox_info['weight']}, {bbox_info['summary']}"
            ).add_to(m)
        
        colormap.caption = "Danger Level (lower votes = more dangerous)"
        colormap.add_to(m)
        
        # Display route information
        st.markdown("### Route Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Start Point**")
            st.write(f"Latitude: {start_lat}")
            st.write(f"Longitude: {start_lon}")
        with col2:
            st.markdown("**End Point**")
            st.write(f"Latitude: {end_lat}")
            st.write(f"Longitude: {end_lon}")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display the map
m.to_streamlit(height=700)


