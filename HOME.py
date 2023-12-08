import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SPROUT",
    page_icon="🌾",
    layout="wide"
)

# Main header
st.title("Welcome to SPROUT 🪴🌾🌱")
st.markdown("### Empowering Plant Enthusiasts and Farmers with Advanced Insights")

# Using columns to create a clean and minimal layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("For Plant Lovers 🪴")
    st.markdown("""
        Discover the secrets of your potted plants with SPROUT:
        - 🌱 Plant Health Analysis
        - 🪴 Personalized Care Tips
        - 🔍 Disease Detection
        - 📈 Growth Tracking
        
        Upload an image of your potted plant and let our AI analyze it for you!
    """)

with col2:
    st.subheader("For Farmers 🌾")
    st.markdown("""
        Transform your farming practices with SPROUT:
        - 🌾 Crop Health Monitoring
        - 🛰️ Satellite Imagery Analysis
        - 💧 Irrigation Optimization
        - 🌍 Land Use Efficiency

        Submit your NDVI images for AI-powered insights, evaluated using TruLens.
    """)

# Footer
st.markdown("---")
st.markdown("### Ready to Grow? Start your journey with SPROUT!")

# Optional: Add navigation to other pages if they exist
st.markdown("Navigate to:")
st.button("Potted Plant Analysis")
st.button("Farmer's NDVI Insights")
