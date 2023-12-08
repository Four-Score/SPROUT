# download_image.py
import streamlit as st
import requests

def download_image(image_url, image_label="Download Image", filename="image.jpg"):
    # Fetch the image from the URL
    img_response = requests.get(image_url)
    if img_response.status_code == 200:
        # Create the download button
        st.download_button(
            label=image_label,
            data=img_response.content,
            file_name=filename,
            mime="image/jpeg"
        )
    else:
        st.error(f"Failed to download image. Status code: {img_response.status_code}")
