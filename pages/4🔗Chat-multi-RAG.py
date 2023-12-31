import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI


from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from utils import get_user_data_from_database, perform_vector_search, create_embeddings_from_image_bytes


import os
import json
from google.cloud import aiplatform
from google.oauth2 import service_account

from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env

# Page configuration
st.set_page_config(page_title="LangChain with Vertex AI", page_icon="🌱")
st.title("SPROUT - Plant 🪴🌱 ")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])


import toml

# Access the credentials
config = st.secrets["google_credentials"]

# Construct a credentials object from the dictionary
credentials = service_account.Credentials.from_service_account_info(config)

# API key
aiplatform.init(project=os.getenv("PROJECT_ID"), location=os.getenv("REGION"), credentials=credentials)

# Use the user_id from session state
user_id = st.session_state.get('user_id')

if user_id:
    # Fetch the user data using the user_id
    user_data = str(get_user_data_from_database(user_id))
    print("type of ud", type(user_data))
    # Use the user_data as needed in your application
else:
    st.error("No user ID found. Please sign up or log in.")
    user_data = ""


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
    description="Search Google for recent results and updated information on famring methods, and practices",
    func=search.run,
)



# Tools (includes query engine)
tools = [GoogleSearch]


chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=chat_model, tools=tools)
executor = AgentExecutor.from_agent_and_tools(agent=chat_agent, tools=tools, memory=memory, return_intermediate_steps=True, handle_parsing_errors=True)


# Chat
if prompt := st.chat_input("Ask a question about farming"):
    with st.chat_message("user"):
        st.write(prompt)
    try:
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)
                st.write("")
    
                # Call the function with the image bytes and a label
                embedding = create_embeddings_from_image_bytes(bytes_data)
                nearest = perform_vector_search(embedding)
                print("nearest  ", nearest, type(nearest))
                st.write("The closest vector search matched for this image are: ", nearest)
                prompt = prompt + "this is info about user's plant(s): " + str(user_data) + " Use the information about user's plant(s) to provide more relevant responses. If the user doesn't specify the plant, ask them to specify a plant first (if there are more than one)." + " This is the result of vector search on image uploaded, indicating the potential plant disease:" + str(nearest)
                print(prompt)
            else:
                prompt = prompt + "this is info about user's plant(s): " + str(user_data) + " Use the information about user's plant(s) to provide more relevant responses. If the user doesn't specify the plant, ask them to specify a plant first (if there are more than one)."
            response = executor(prompt, callbacks=[st_cb])
            st.write(response["output"])
    except Exception as e:
        st.error("Feature is unable to work because maximum Streamlit resource limit has been reached.")

