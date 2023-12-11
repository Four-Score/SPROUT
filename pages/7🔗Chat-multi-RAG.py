import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper, OpenWeatherMapAPIWrapper
from llama_hub.tools.weather import OpenWeatherMapToolSpec
from llama_index.embeddings import GooglePaLMEmbedding
import os
import json
from google.cloud import aiplatform
import vertexai
from google.oauth2 import service_account
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import os
import requests
from io import BytesIO
from PIL import Image
import torch
from llama_index.multi_modal_llms import ReplicateMultiModal
from llama_index.multi_modal_llms.replicate_multi_modal import REPLICATE_MULTI_MODAL_LLM_MODELS
from langchain.tools import BaseTool
import replicate

# Imports main tools:
from trulens_eval import TruChain, Feedback, Tru
from trulens_eval import feedback, Feedback
tru = Tru()
tru.reset_database()

import os
import json
from google.cloud import aiplatform
import vertexai
from google.oauth2 import service_account
from classify import make_prediction
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env

# Page configuration
st.set_page_config(page_title="LangChain with Vertex AI", page_icon="🌱")
st.title("SPROUT - Farm 🌾🌱 ")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])


import toml

# Access the credentials
#config = st.secrets["google_credentials"]

# Construct a credentials object from the dictionary
#credentials = service_account.Credentials.from_service_account_info(config)


# API key
#aiplatform.init(project=os.getenv("PROJECT_ID_CODE"), location=os.getenv("REGION"), credentials=credentials)




# Create a Vertex AI agent
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    
avatars = {"human": "user", "ai": "assistant"}

for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        st.write(msg.content)

# Vertex AI model and tools
llm = VertexAI()
# Uses text - bison model

chat_model = ChatVertexAI(llm=llm)
# Uses chat - bison model


# Set your Replicate API token
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

# Initialize the Replicate client
client = replicate.Client()

if 'soil_data' in st.session_state:
    soil_data = str(st.session_state['soil_data'])
    # Use the soil_data as needed
else:
    st.warning("Soil data is not available. Please go back and fetch the soil data first.")

if 'user_location' in st.session_state:
    # Do something with the user's location
    location = st.session_state['user_location']

def preprocess_ndvi_image(image):
    """
    Analyze the NDVI image using a multi-modal LLM (like Fuyu-8B) and return the analysis.
    """
    # Convert the uploaded file to a format suitable for the multi-modal LLM
    image = Image.open(image)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Call the multi-modal LLM model
    response = client.predictions.create(
        version="fuyu-8b",  # Replace with the specific model version you're using
        input={
            "image": buffer,
            "prompt": "Describe the vegetation health and areas of concern in this NDVI image:"
        }
    )

    # Extract and return the analysis from the response
    analysis = response["output"]
    return analysis



search = GoogleSearchAPIWrapper()
weather = OpenWeatherMapAPIWrapper()

GoogleSearch = Tool(
    name="Google Search",
    description="Search Google for recent results and updated information on farming methods, practices, and advice.",
    func=search.run,
)

WeatherData = Tool(
    name="Weather Forecast",
    description="Get the weather forecast for optimized farming methods and practices.",
    func=weather.run,
)

# Tools (includes query engine)
tools = [GoogleSearch]


chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=chat_model, tools=tools)
executor = AgentExecutor.from_agent_and_tools(agent=chat_agent, tools=tools, memory=memory, return_intermediate_steps=True, handle_parsing_errors=True)

# Chat interaction
# Check for image upload
uploaded_ndvi_image = st.file_uploader("Upload an NDVI image", type=["jpg", "jpeg", "png"])
image_analysis = ""
if uploaded_ndvi_image is not None:
    # Preprocess the image with the multi-modal LLM
    image_analysis = preprocess_ndvi_image(uploaded_ndvi_image)
    
hugs = feedback.Huggingface()

f_lang_match = Feedback(hugs.language_match).on_input_output()
pii = Feedback(hugs.pii_detection).on_output()
pos = Feedback(hugs.positive_sentiment).on_output()
tox = Feedback(hugs.toxic).on_output() 


tru_recorder = TruChain(executor,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match, pii, pos, tox])

with tru_recorder as recording:
    # Chat
    if prompt := st.chat_input("Ask a question about planting"):
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            combined_prompt = f"NDVI Image Analysis: {image_analysis}\n\nUser Query: {prompt}\n\nSoil Data: {soil_data}\n\nFarm Location: {location}"
            response = executor(prompt, callbacks=[st_cb])
            st.write(response["output"])

# Displaying Results
with st.expander("Detailed Evaluation Results"):
    records, feedback = tru.get_records_and_feedback(app_ids=[])
    st.dataframe(records)
    
with st.container():
    st.header("Evaluation")    
    st.dataframe(tru.get_leaderboard(app_ids=[]))
    st.dataframe(feedback)
