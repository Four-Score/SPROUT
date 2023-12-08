import streamlit as st

def potted_plant_front_page():
    st.title("🪴 Potted Plant Insights")

    st.markdown("""
        🪴 **Welcome to the Potted Plant Insights section of SPROUT.** Dive into personalized plant care and exploration.

        🪴 **Explore our offerings:**
        - **Form**: Start by filling out a detailed form about your plant (find this in the sidebar).
        - **Chat Systems**: 
            - *Multi-modal RAG Chat*: Engage with our retrieval-augmented generation chat for in-depth insights.
            - *Classification + Chat*: For specific queries, use our classification-based chat system.
        - **Evaluation**: Compare and evaluate the insights from both chats using TruEra.

        🪴 *Check out 'Form' from the sidebar to begin sharing details about your potted plant.*
    """)

    st.markdown("---")
    st.markdown("*Navigate and explore using the sidebar.*")

if __name__ == "__main__":
    potted_plant_front_page()
