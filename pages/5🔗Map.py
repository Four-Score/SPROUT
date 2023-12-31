import streamlit as st
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import requests
import json
import os
import re
from satellite_image import get_most_recent_images, display_images_and_stats
from soil_data import process_soil_data
from soil_data import get_soil_data
from dotenv import load_dotenv

load_dotenv()  # This loads the environment variables from .env file


# Assume the API key is set in your environment variables
api_key = os.getenv('AGROMONITORING_API_KEY')

# Function to create a polygon via the API, with an option to allow duplicates
def create_polygon(api_key, geo_json_data, polygon_name, allow_duplicates=False):
    api_url = f'https://api.agromonitoring.com/agro/1.0/polygons?appid={api_key}'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "name": polygon_name,
        "geo_json": geo_json_data
    }
    # Add the 'duplicated' flag to the request parameters if duplicates are allowed
    params = {'duplicated': 'true'} if allow_duplicates else {}

    response = requests.post(api_url, headers=headers, params=params, json=payload)
    if response.status_code == 201:
        polygon_id = response.json()['id']
        print(f"Polygon created with ID: {polygon_id}")  # Print the Polygon ID
        return polygon_id
    else:
        response_data = response.json()
        error_message = response_data.get('error', {}).get('message', '')
        if "UnprocessableEntityError" in error_message:
            area_match = re.search(r"Area of the polygon - ([\d.]+) ha", error_message)
            if area_match:
                area = area_match.group(1)
                st.info(f"The area of the polygon is currently {area} hectares. Please ensure the area is between 1 to 3000 hectares.")
            else:
                st.info("The area of the polygon is outside the acceptable range. Please adjust the size of your polygon.")
        else:
            st.info(f"Please draw your polygon again: {response.text}")
        return None

def search_satellite_images(api_key, polygon_id):
    # Use current time for both start and end to fetch the most recent images
    end_timestamp = int(datetime.now().timestamp())  # Current timestamp
    start_timestamp = end_timestamp - ( 60* 24 * 3600)  # 30 days ago

    api_url = f'http://api.agromonitoring.com/agro/1.0/image/search?appid={api_key}&polyid={polygon_id}&start={start_timestamp}&end={end_timestamp}'
    response = requests.get(api_url)
    if response.status_code == 200:
        imagery_data = response.json()
        if not imagery_data:  # No images found
            print("API did not find any images, response was empty.")
        return imagery_data
    else:
        print(f"API error occurred: {response.status_code}, {response.text}")  # Print out the error
        return []

# Streamlit app logic
def main():
    st.title("🌱Sprout's Field Watch: Your Eye in the Sky")
    # Introductory text and instructions
    st.markdown("""
        Welcome to the Agricultural Monitoring App. Follow these steps to view satellite imagery and soil data:
        1. Draw a polygon on the map and click on the export button to save the GeoJSON file.
        2. Enter a name for your polygon.
        3. Upload the exported GeoJSON file.
        4. Click 'Fetch Satellite Images & Soil Data' to view the information,you can also download the NDVI image.
    """)
    st.subheader("Enter your location")
    user_location = st.text_input("(city, country):")
    if user_location:  # Check if user has entered a location
        st.session_state['user_location'] = user_location  # Save the location in session state

    # Map initialization
    st.subheader("Draw a Polygon on the Map")
    m = folium.Map(location=[24.8607, 67.0011], zoom_start=10)  # Karachis coordinates
    draw = folium.plugins.Draw(export=True)
    draw.add_to(m)
    folium_static(m)

    # Polygon name input
    st.subheader("Enter name of your polygon")
    polygon_name = st.text_input("Polygon name:")

    # File uploader
    st.subheader("Upload the GeoJSON File")
    uploaded_file = st.file_uploader("Choose the exported GeoJSON file", type=['geojson'])

    # Fetch satellite images button
    st.subheader("Click the button below to fetch image and soil data")
    if st.button("Fetch Satellite Images Soil data"):
        if uploaded_file and polygon_name:
            geo_json_data = json.load(uploaded_file)
            polygon_id = create_polygon(api_key, geo_json_data, polygon_name, allow_duplicates=True)
            if polygon_id:
                # Now pass this polygon_id to the satellite_images.py functions
                most_recent_image_data = get_most_recent_images(polygon_id, api_key)
                if most_recent_image_data:
                    display_images_and_stats(most_recent_image_data,api_key)
                else:
                    st.error("Could not find satellite images for the given polygon ID.")
                # Fetch and display soil data

                soil_data = get_soil_data(polygon_id, api_key)
                if soil_data:
                    processed_soil_data = process_soil_data(soil_data)
                    if process_soil_data:
                        st.session_state['soil_data'] = processed_soil_data

                        st.subheader("Soil Data")
                        for key, value in processed_soil_data.items():
                            st.text(f"{key}: {value}")
                    else:
                        st.error("soil data not available or key is missing")
                
                else:
                     st.error(f"Failed to fetch soil data: {soil_data.text}")              
            else:
                st.warning("A polygon of the specified area couldnot be created")
        else:
            st.warning("Please upload a GeoJSON file and enter a name for the polygon.")

if __name__ == "__main__":
    main()
