import streamlit as st

def farm_ndvi_front_page():
    st.title("🌾 Farm NDVI Insights")

    st.markdown("""
        🌾 **Welcome to the Farm NDVI Insights section of SPROUT.** 
        Empowering farmers with advanced crop management tools.

        🌾 **Your journey here includes:**
        - **Interactive Map**: Start by exploring your farm's NDVI data and soil analysis (available in the sidebar).
        - **Chat Systems**: 
            - *Multi-modal RAG Chat*: Get comprehensive farming advice and insights.
            - *Vision-based Chat*: Focus on image-based queries and specific agricultural insights.
        - **Evaluation**: Assess the responses from both chat systems through TruEra.

        🌾 *Select 'Interactive Map' from the sidebar to view detailed information about your farm.*
    """)

    st.markdown("---")
    st.markdown("*Use the sidebar to navigate through the features.*")

if __name__ == "__main__":
    farm_ndvi_front_page()
