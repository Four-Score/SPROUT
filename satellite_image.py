import requests
from datetime import datetime
import streamlit as st
import pandas as pd

# Function to fetch the most recent satellite images for a given polygon_id
def get_most_recent_images(polygon_id, api_key):
    # Extend the search to the past two years
    end_timestamp = int(datetime.utcnow().timestamp())
    start_timestamp = end_timestamp - (2 * 365 * 24 * 60 * 60)  # 2 years ago in seconds
    
    # Set up the API endpoint with the start and end timestamp
    api_url = f"http://api.agromonitoring.com/agro/1.0/image/search?start={start_timestamp}&end={end_timestamp}&polyid={polygon_id}&appid={api_key}"
    
    # Make the API request
    response = requests.get(api_url)
    if response.status_code == 200:
        images = response.json()
        if not images:
            return None  # No images found
        # Sort images by date and select the most recent one
        most_recent_image = max(images, key=lambda x: x['dt'])
        return most_recent_image
    else:
        st.error(f"Failed to fetch satellite images: {response.text}")
        return None


def display_images_and_stats(image_data, api_key):
    if image_data:
        col1, col2 = st.columns([1, 3])  # Adjust the ratio as per your UI needs

        with col1:
            # Display the NDVI image
            ndvi_image_url = image_data['image']['ndvi']
            st.image(ndvi_image_url, caption="Most Recent NDVI Image", width=150)
            
            # Add a button to download the NDVI image
            if ndvi_image_url:
                # Fetch the image from the URL
                img_response = requests.get(ndvi_image_url)
                if img_response.status_code == 200:
                    # Use the download button to download the image
                    st.download_button(
                        label="Download NDVI Image",
                        data=img_response.content,
                        file_name="ndvi_image.jpg",
                        mime="image/jpg"
                    )
                else:
                    st.error("Failed to download NDVI image.")
        with col2:
            # Create an empty DataFrame to store the statistics
            stats_df = pd.DataFrame()

            # Iterate over each index type to fetch and display its statistics
            for index_type in ['ndvi', 'evi', 'evi2', 'nri', 'dswi', 'ndwi']:
                # Correctly construct the statistics URL
                base_url, _ = image_data['stats'][index_type].split('?')
                stat_url = f"{base_url}?appid={api_key}"
                
                # Print the URL for debugging purposes
                print(f"Constructed URL for {index_type} statistics: {stat_url}")
                
                # Make the request to the API
                response = requests.get(stat_url)
                if response.status_code == 200:
                    stats = response.json()
                    # Append the stats to the DataFrame, rounding to two decimal places
                    stats_df[index_type.upper()] = [
                        round(stats.get('max', 'N/A'), 2),
                        round(stats.get('mean', 'N/A'), 2),
                        round(stats.get('median', 'N/A'), 2),
                        round(stats.get('min', 'N/A'), 2),
                        round(stats.get('std', 'N/A'), 2),
                        round(stats.get('num', 'N/A'), 2)
                    ]
                else:
                    st.error(f"Failed to load {index_type.upper()} data: {response.text}")

            # Set the index of the DataFrame for better readability
            stats_df.index = ['Max', 'Mean', 'Median', 'Min', 'Standard Deviation', 'Number of Pixels']

            # Display the DataFrame as a table
            st.table(stats_df.T)  # Transpose for better layout
