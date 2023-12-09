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
from google.oauth2 import service_account
from google.cloud import aiplatform
import os
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

st.set_page_config(page_title="LangChain with Vertex AI", page_icon="🌱")
st.title("SPROUT - Farm 🌾🌱")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
plants_info = []

# Fetch the user_id from the session state
user_id = st.session_state.get('user_id')
if user_id:
    # Fetch the user data from the database
    user_data = get_user_data_from_database(user_id)
    
    # Parse the user data to extract the plant data
    try:
        plants_info = json.loads(user_data)
        # Ensure we have a list of dicts, each dict containing a plant's data
        if not isinstance(plants_info, list) or not all(isinstance(plant, dict) for plant in plants_info):
            raise ValueError("User data is not in the expected format")
        
        # Generate a list of plant names for the radio buttons
        plant_names = [plant['name'] for plant in plants_info if 'name' in plant]
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error parsing user data: {e}")
        plant_names = []
else:
    st.error("No user ID found. Please sign up or log in.")
    plant_names = []

# If no plant names found, display an error
if not plant_names:
    st.error("No plant data available. Please add plant data.")
else:
    # Display radio buttons for plant names
    selected_plant_name = st.radio("Select a plant:", plant_names)
    
    # Get the full data of the selected plant
    selected_plant_data = next((plant for plant in plants_info if plant['name'] == selected_plant_name), None)


config = st.secrets["google_credentials"]
credentials = service_account.Credentials.from_service_account_info(config)

aiplatform.init(project=os.getenv("PROJECT_ID_CODE"), location=os.getenv("REGION"), credentials=credentials)

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

search = GoogleSearchAPIWrapper()
GoogleSearch = Tool(
    name="Google Search",
    description="Search Google for recent results and updated information on farming methods and practices",
    func=search.run,
)
tools = [GoogleSearch]

chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=chat_model, tools=tools)
executor = AgentExecutor.from_agent_and_tools(agent=chat_agent, tools=tools, memory=memory, return_intermediate_steps=True, handle_parsing_errors=True)

f prompt := st.chat_input(f"Ask a question about {selected_plant_name}:"):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Augment the prompt with specific plant data if a plant is selected
        if selected_plant_data:
            # Convert selected plant data to JSON string
            plant_info = json.dumps(selected_plant_data)
            augmented_prompt = f"{prompt} [Plant info: {plant_info}]"
        else:
            augmented_prompt = prompt

        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)
            st.write("")

            # Call the function with the image bytes and a label
            embedding = encode_images_to_embeddings(image_bytes=bytes_data, label='Uploaded Image')
            nearest = str(findneighbor_sample(embedding['image_embedding']))
            print("nearest  ", nearest, type(nearest))
            prompt = prompt + "this is info about user's plant(s): " + user_data + " this is the result of vector search on image uploaded, indicating the potential plant disease:" + nearest
            print(prompt)
        else:
            prompt = prompt

        response = executor(augmented_prompt, callbacks=[st_cb])
        st.write(response["output"])
