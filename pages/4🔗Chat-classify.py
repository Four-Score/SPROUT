import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI


from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper



import os
import json
from google.cloud import aiplatform
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
config = st.secrets["google_credentials"]

# Construct a credentials object from the dictionary
credentials = service_account.Credentials.from_service_account_info(config)



# API key
aiplatform.init(project=os.getenv("PROJECT_ID_CODE"), location=os.getenv("REGION"), credentials=credentials)




# Create a Vertex AI agent
#msgs = StreamlitChatMessageHistory()
# Function to get a unique session key based on the current page
def get_session_key():
    # Assuming you have some logic to determine the current page
    current_page = st.session_state.current_page
    return f"chat_history_{current_page}"

# Initialize or retrieve chat history specific to the current page
session_key = get_session_key()
if session_key not in st.session_state:
    st.session_state[session_key] = StreamlitChatMessageHistory()
msgs = st.session_state[session_key]

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




# TOOLS
#Search
search = GoogleSearchAPIWrapper()
GoogleSearch = Tool(
    name="Google Search",
    description="Search Google for recent results and updated information on famring methods, and practices",
    func=search.run,
)



# Tools (includes query engine)
tools = [GoogleSearch]


chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=chat_model, tools=tools)
executor = AgentExecutor.from_agent_and_tools(agent=chat_agent, tools=tools, memory=memory, return_intermediate_steps=True, handle_parsing_errors=True)





# Chat
if prompt := st.chat_input("Ask a question about planting"):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)
            st.write("")

            # Call the classifier with the image bytes
            predictions = make_prediction(bytes_data)
            prediction_text = "This is the result of the classifier on the image uploaded, indicating the potential plant status: " + ", ".join([str(prediction) for prediction in predictions])
            prompt = prompt + " " + prediction_text
            print(prompt)
        else:
            prompt = prompt
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
