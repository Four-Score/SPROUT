import streamlit as st
import json
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from utils import get_user_data_from_database, make_prediction, embed_info
from google.oauth2 import service_account
from google.cloud import aiplatform 
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(page_title="LangChain with Vertex AI", page_icon="🌱")
st.title("SPROUT - Plant 🪴🌱")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Assuming the user_id is retrieved from the user's login session
user_id = "user123"  # Replace this with the actual logic to retrieve the logged-in user's ID
user_data = get_user_data_from_database(user_id)

# Extract plant names from user_data
try:
    plants_data = json.loads(user_data)
    plant_names = [plant.get("name", "Unnamed Plant") for plant in plants_data]
except json.JSONDecodeError:
    st.error("Error parsing user data")
    plant_names = []

selected_plant = st.selectbox("Select a plant:", plant_names)

# Initialize LangChain with Vertex AI
config = st.secrets["google_credentials"]
credentials = service_account.Credentials.from_service_account_info(config)
aiplatform.init(project=os.getenv("PROJECT_ID_CODE"), location=os.getenv("REGION"), credentials=credentials)

# Create a Vertex AI agent
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()

avatars = {"human": "user", "ai": "assistant"}

for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        st.write(msg.content)

llm = VertexAI()
chat_model = ChatVertexAI(llm=llm)

# Tools and Agent Setup
search = GoogleSearchAPIWrapper()
GoogleSearch = Tool(
    name="Google Search",
    description="Search Google for recent results and updated information on farming methods and practices",
    func=search.run,
)
tools = [GoogleSearch]

chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=chat_model, tools=tools)
executor = AgentExecutor.from_agent_and_tools(agent=chat_agent, tools=tools, memory=memory, return_intermediate_steps=True, handle_parsing_errors=True)

# Chat
if prompt := st.chat_input("Ask a question about planting"):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Augment the prompt with selected plant information
        augmented_prompt = f"{prompt} [Selected plant: {selected_plant}]"

        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)
            predictions = make_prediction(bytes_data)
            prediction_text = "Classifier result: " + ", ".join([str(prediction) for prediction in predictions])
            augmented_prompt += " " + prediction_text

        response = executor(augmented_prompt, callbacks=[st_cb])
        st.write(response["output"])
