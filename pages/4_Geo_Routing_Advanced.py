import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import requests
import polyline
import os
from dotenv import load_dotenv

# --- Load API Key ---
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# --- Cached Data Loaders ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/bay_area_providers_routing.csv")
    df["urgency_score"] = df.get("Overall Rating", 1).fillna(0)
    df = df.sort_values("urgency_score", ascending=False).head(20).reset_index(drop=True)
    return df

@st.cache_data
def get_distance_matrix(locations, batch_size=10):
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    for i in range(0, n, batch_size):
        origin_str = "|".join([f"{lat},{lon}" for lat, lon in locations[i:i+batch_size]])
        for j in range(0, n, batch_size):
            dest_str = "|".join([f"{lat},{lon}" for lat, lon in locations[j:j+batch_size]])
            url = (
                f"https://maps.googleapis.com/maps/api/distancematrix/json"
                f"?units=imperial&origins={origin_str}&destinations={dest_str}&key={GOOGLE_MAPS_API_KEY}"
            )
            response = requests.get(url).json()
            if response.get("status") != "OK":
                st.error(f"API Error: {response.get('error_message')}")
                return matrix
            for oi, row in enumerate(response.get("rows", [])):
                for di, element in enumerate(row.get("elements", [])):
                    try:
                        miles = element["distance"]["value"] / 1609.34
                    except (KeyError, TypeError):
                        miles = 999
                    matrix[i + oi][j + di] = miles
    return matrix

@st.cache_data
def get_road_route(start, end):
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={start[0]},{start[1]}"
        f"&destination={end[0]},{end[1]}"
        f"&mode=driving&key={GOOGLE_MAPS_API_KEY}"
    )
    response = requests.get(url).json()
    if response.get("status") != "OK":
        st.error(f"Google API error: {response.get('status')} - {response.get('error_message', 'No details')}")
        return []
    try:
        poly = response["routes"][0]["overview_polyline"]["points"]
        return polyline.decode(poly)
    except Exception as e:
        st.error(f"Polyline decode error: {e}")
        return []

# --- TSP Solver ---
def solve_tsp_nearest_neighbor(matrix):
    n = len(matrix)
    visited = [False] * n
    path = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = path[-1]
        next_city = min(
            [(i, matrix[last][i]) for i in range(n) if not visited[i]],
            key=lambda x: x[1]
        )[0]
        path.append(next_city)
        visited[next_city] = True
    return path

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🚗 Optimized Facility Route with Real Road Paths")
st.markdown(
    "*Data sourced from [CMS](https://data.cms.gov/), filtered for Bay Area nursing facilities. "
    "We cluster providers using KMeans into 4 visit groups (days), then optimize daily visit routes "
    "with the Nearest Neighbor heuristic for the Traveling Salesman Problem (TSP), using real road "
    "distances from the Google Maps API.*"
)

df = load_data()

# Clustering for Multi-Day Route Splits
kmeans = KMeans(n_clusters=4, random_state=42).fit(df[['Latitude', 'Longitude']])
df['cluster'] = kmeans.labels_

# Day Selector
day_map = {i: f"Day {i+1}" for i in sorted(df['cluster'].unique())}
day_selection = st.selectbox("📅 Select Day", options=list(day_map.keys()), format_func=lambda x: day_map[x])
day_df = df[df['cluster'] == day_selection].reset_index(drop=True)

# if st.checkbox("🔧 Use development mode (only 8 HCPs)", value=True):
#     day_df = day_df.head(8)

# Distance & TSP Optimization
locations = list(zip(day_df["Latitude"], day_df["Longitude"]))
distance_matrix = get_distance_matrix(locations)
route_order = solve_tsp_nearest_neighbor(distance_matrix)
day_df = day_df.iloc[route_order].reset_index(drop=True)

# Map Initialization
min_lat, max_lat = day_df["Latitude"].min(), day_df["Latitude"].max()
min_lon, max_lon = day_df["Longitude"].min(), day_df["Longitude"].max()
center_lat, center_lon = day_df["Latitude"].mean(), day_df["Longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap") if min_lat == max_lat else folium.Map(tiles="OpenStreetMap")
if min_lat != max_lat:
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

# Marker and Route Visualization
# Marker and Route Visualization (with START/END clearly marked)
marker_cluster = MarkerCluster().add_to(m)
color_map = {0: 'blue', 1: 'green', 2: 'red', 3: 'purple'}

for idx, row in day_df.iterrows():
    popup_html = (
        f"<b>{row['Provider Name']}</b><br>"
        f"Address: {row['Provider Address']}, {row['City/Town']}<br>"
        f"Rating: {row['urgency_score']}<br>"
        f"Beds: {row['Number of Certified Beds']}<br>"
        f"Residents/day: {row['Average Number of Residents per Day']}<br>"
        f"<a href='https://www.google.com/maps/search/?api=1&query={row['Latitude']},{row['Longitude']}' target='_blank'>Open in Google Maps</a>"
    )

    if idx == 0:
        icon = folium.Icon(color='green', icon='play', prefix='fa')
        tooltip = f"🟢 START: {row['Provider Name']}"
    elif idx == len(day_df) - 1:
        icon = folium.Icon(color='red', icon='stop', prefix='fa')
        tooltip = f"🔴 END: {row['Provider Name']}"
    else:
        icon = folium.Icon(color=color_map.get(row['cluster'], 'gray'))
        tooltip = row["Provider Name"]

    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=tooltip,
        icon=icon
    ).add_to(marker_cluster)


# Draw actual road paths between points (using reordered day_df)
for i in range(len(day_df) - 1):
    start_latlon = (day_df.loc[i, "Latitude"], day_df.loc[i, "Longitude"])
    end_latlon = (day_df.loc[i + 1, "Latitude"], day_df.loc[i + 1, "Longitude"])
    decoded_path = get_road_route(start_latlon, end_latlon)
    if decoded_path:
        folium.PolyLine(decoded_path, color="orange", weight=4, opacity=0.8).add_to(m)
    else:
        st.warning(f"No route found between: {day_df.loc[i, 'Provider Name']} ➡️ {day_df.loc[i + 1, 'Provider Name']}")




# Render Map
st.subheader("🗺️ Real-Route Optimized Driving Path")
st_folium(m, width=1100, height=600)

# Total Distance
total_distance = sum([
    distance_matrix[route_order[i]][route_order[i+1]] for i in range(len(route_order) - 1)
])
st.success(f"**Total Distance Traveled:** {total_distance:.2f} miles")

# Downloadable CSV
st.download_button("📥 Download Route CSV", data=day_df.to_csv(index=False), file_name="optimized_route.csv")

# Step-by-Step Table
st.markdown("## 📍 Step-by-Step Route Breakdown")
route_rows = []
for i in range(len(day_df)):
    row = day_df.iloc[i]
    prev_dist = 0 if i == 0 else distance_matrix[route_order[i-1]][route_order[i]]
    route_rows.append({
        "Stop #": i + 1,
        "Provider Name": row["Provider Name"],
        "Address": f"{row['Provider Address']}, {row['City/Town']}, {row['State']} {row['ZIP Code']}",
        "Distance from Previous (miles)": round(prev_dist, 2)
    })
st.dataframe(pd.DataFrame(route_rows), use_container_width=True)

# Explanation Panel
with st.expander("ℹ️ How Routes Are Optimized"):
    st.markdown("""
    - Facilities are grouped into 4 clusters using **KMeans clustering**, suggesting logical groupings for **multi-day visits** based on proximity.
    - Each day's route is optimized with a **Nearest Neighbor** TSP heuristic to reduce driving time.
    - Road paths come from **Google Directions API** for real-world accuracy.
    - **Marker clustering** is enabled for map clarity—zoom in to see individual points.
    """)
