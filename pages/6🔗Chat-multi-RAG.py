import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain


from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import OpenWeatherMapAPIWrapper
from llama_hub.tools.weather import OpenWeatherMapToolSpec




from llama_index.embeddings import GooglePaLMEmbedding


import os
import json
from google.cloud import aiplatform
import vertexai
from google.oauth2 import service_account

from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env

# Page configuration
st.set_page_config(page_title="LangChain with Vertex AI", page_icon="🌱")
st.title("SPROUT - Farm 🌾🌱 ")




import toml

# Access the credentials
config = st.secrets["google_credentials"]

# Construct a credentials object from the dictionary
credentials = service_account.Credentials.from_service_account_info(config)


# API key
aiplatform.init(project=os.getenv("PROJECT_ID"), location=os.getenv("REGION"), credentials=credentials)




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




# TOOLS
#Search
search = GoogleSearchAPIWrapper()
GoogleSearch = Tool(
    name="Google Search",
    description="Search Google for recent results and updated information on famring methods, and practices and advice custom to the user as well as searching overall climatic, financial and economic conditions of the user's location",
    func=search.run,
)

#Climate Forecast
weather = OpenWeatherMapAPIWrapper()
WeatherData = Tool(
    name="Weather Forecast",
    description="Get the weather forecast for deciding optimized farming methods and practices custom to the user and their location",
    func=weather.run,
)



# Tools (includes query engine)
tools = [GoogleSearch, WeatherData]

chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=chat_model, tools=tools)
executor = AgentExecutor.from_agent_and_tools(agent=chat_agent, tools=tools, memory=memory, return_intermediate_steps=True, handle_parsing_errors=True)


# Chat
if prompt := st.chat_input("Ask a question about farming"):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
