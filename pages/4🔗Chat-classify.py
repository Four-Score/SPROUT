import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI


from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from utils import get_user_data_from_database
import os
import json
from google.cloud import aiplatform
from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env

# Page configuration
st.set_page_config(page_title="LangChain with Vertex AI", page_icon="🌱")
st.title("SPROUT - Plant 🌾🌱 ")
# Load the model for image classification
# File uploader for image classification
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'my_model.hdf5'  


model = load_model(model_path)

st.write("# Flower Classification")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # Ensure these match your model's classes

def import_and_predict(image_data, model):
    size = (180, 180)    
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)  # Updated for new Pillow version
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure color format matches your model's training
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', width=250)  # Adjust the width as needed

    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")


import toml
# API key and Vertex AI initialization
#import toml

# Access the credentials


#Access the credentials
config = st.secrets["google_credentials"]# Convert the string back to a JSON object
#Construct a credentials object from the dictionary
credentials = service_account.Credentials.from_service_account_info(config)

# API key
aiplatform.init(project=os.getenv("PROJECT_ID_CODE"), location=os.getenv("REGION"), credentials=credentials)

# Use the user_id from session state
user_id = st.session_state.get('user_id')

if user_id:
    # Fetch the user data using the user_id
    user_data = str(get_user_data_from_database(user_id))
    print("type of ud", type(user_data))
    # Use the user_data as needed in your application
else:
    st.error("No user ID found. Please sign up or log in.")


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



# Chat and image classification integration
if prompt := st.chat_input("Ask a question about planting"):
    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            if file is not None:
                image = Image.open(file)
                st.image(image, caption='Uploaded Image.', width=250)

                prompt = "This is the user's query:" + prompt + " " + predicted_class
            else:
                prompt = "This is the user's query:" + prompt

            response = executor(prompt, callbacks=[st_cb])
            st.write(response["output"])
    except Exception as e:
        st.error("Feature is unable to work because maximum Streamlit resource limit has been reached.")



