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
from utils import get_user_data_from_database
from classify import make_prediction
from google.oauth2 import service_account
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(page_title="LangChain with Vertex AI", page_icon="🌱")
st.title("SPROUT - Plant 🪴🌱")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Function to display plant selection
def display_plant_selection(user_data):
    if user_data:
        user_plants = json.loads(user_data).get("plants", [])
        selected_plants = st.multiselect("Select your plants:", user_plants)
        return selected_plants
    return []

# Use the user_id from session state
user_id = st.session_state.get('user_id')
if user_id:
    user_data = get_user_data_from_database(user_id)
    selected_plants = display_plant_selection(user_data)
else:
    user_data = ""
    selected_plants = []

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
        if selected_plants:
            plant_info = " User's selected plants: " + ", ".join(selected_plants)
            prompt += plant_info

        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)
            predictions = make_prediction(bytes_data)
            prediction_text = "Classifier result: " + ", ".join([str(prediction) for prediction in predictions])
            prompt += " " + prediction_text

        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
