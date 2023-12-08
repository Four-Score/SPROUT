import streamlit as st
import os
import json
from google.cloud import aiplatform
from google.oauth2 import service_account
from dotenv import load_dotenv
import replicate
from llama_index.multi_modal_llms import ReplicateMultiModal
from llama_index.schema import ImageDocument

from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

# Load environment variables
load_dotenv()

# API key and Vertex AI initialization
credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
credentials_dict = json.loads(credentials_json)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)
aiplatform.init(project=os.getenv("PROJECT_ID"), location=os.getenv("REGION"), credentials=credentials)

# Page configuration
st.set_page_config(page_title="LangChain with Vertex AI", page_icon="🌱")
st.title("SPROUT - Farm 🌾🌱")

# Initialize message history and memory
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()

# Initialize tools and agent
search = GoogleSearchAPIWrapper()
GoogleSearch = Tool(
    name="Google Search",
    description="Search Google for recent results and updated information on farming methods, practices, and advice.",
    func=search.run,
)
tools = [GoogleSearch]

llm = VertexAI()
chat_model = ChatVertexAI(llm=llm)
chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=chat_model, tools=tools)
executor = AgentExecutor.from_agent_and_tools(agent=chat_agent, tools=tools, memory=memory, return_intermediate_steps=True, handle_parsing_errors=True)

# Set up your Replicate API token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Initialize the Replicate client
client = replicate.Client()

import tempfile
import shutil
def display_spinner():
    with st.spinner('Processing image... Please wait'):
        return
def preprocess_image(uploaded_image):
    """
    Analyze the image using a multi-modal LLM (like Fuyu-8B) and return the analysis.
    """
    try:
        with st.spinner("Processing image... Please wait"):
            print("Starting image processing...")

            # Save the uploaded image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                shutil.copyfileobj(uploaded_image, tmp_file)
                tmp_file_path = tmp_file.name
                print(f"Image saved to temporary file: {tmp_file_path}")

            # Prepare the image document with the file path
            image_document = ImageDocument(image_path=tmp_file_path)
            print("ImageDocument created successfully.")

            # Initialize the MultiModal LLM model
            multi_modal_llm = ReplicateMultiModal(
                model="lucataco/fuyu-8b:42f23bc876570a46f5a90737086fbc4c3f79dd11753a28eaa39544dd391815e9",
                max_new_tokens=500,
                temperature=0.1,
                num_input_files=1,
                top_p=0.9,
                num_beams=1,
                repetition_penalty=1,
            )
            print("MultiModal LLM initialized successfully.")

            # Inside the preprocess_image function

            # Enhanced prompt for image analysis
            enhanced_prompt = """
                Analyze the provided NDVI image of a farm. The image uses a green palette, 
                common in NDVI imagery, with control points at (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1). 
                Based on this, assess the vegetation health, identify areas with potential issues, 
                and provide insights on crop growth patterns and potential improvements. 
                Consider factors such as crop density, water stress, and nutrient levels.
                """
            

            # Get the response from the multi-modal LLM
            mm_resp = multi_modal_llm.complete(
                prompt=enhanced_prompt,
                image_documents=[image_document],
            )

            print("Response received from MultiModal LLM.")

            # Directly access the response data
            # Modify this part based on the actual structure of mm_resp
            if mm_resp and hasattr(mm_resp, 'text'):
                analysis = mm_resp.text
                print("Analysis extracted successfully.")
            else:
                print("No valid response received.")
                analysis = ""

            # Optionally delete the temporary file after processing
            os.remove(tmp_file_path)

        return analysis

    except Exception as e:
        print(f"Error during image processing: {e}")
        st.error(f"Failed to process the image. Error: {e}")
        return ""



#User Interface
uploaded_image = st.file_uploader("Upload an NDVI image of your farm", type=["jpg", "jpeg", "png"])
image_analysis = ""

if uploaded_image is not None:
    image_analysis = preprocess_image(uploaded_image)
    if image_analysis:
        with st.expander("Image Analysis Result:"):
            st.write(image_analysis)

# Chat interaction
prompt = st.text_input("Ask a question or request specific advice about your farm:")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    combined_prompt = f"{image_analysis}\n\nUser Query: {prompt}" if image_analysis else f"User Query: {prompt}"

    with st.spinner('Generating response...'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(combined_prompt, callbacks=[st_cb])
        
        with st.chat_message("assistant"):
            st.write(response["output"])