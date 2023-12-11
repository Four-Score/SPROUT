from llama_index.embeddings import HuggingFaceEmbedding
import hashlib
import json, time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient
import os
import os
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env
CLUSTER_ENDPOINT= os.getenv("CLUSTER_ENDPOINT") # cluster endpoint
TOKEN=os.getenv("TOKEN") # Set your token
COLLECTION_NAME="user_data" # Set your collection name


connections.connect(
  alias='default', 
  #  Public endpoint obtained from Zilliz Cloud
  uri=CLUSTER_ENDPOINT,
  # API key or a colon-separated cluster username and password
  token=TOKEN, 
)

client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
collection = Collection(uri=CLUSTER_ENDPOINT, token=TOKEN, name=COLLECTION_NAME)

  

def embed_info(info):
    # loads BAAI/bge-small-en-v1.5
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    embeddings = embed_model.get_text_embedding(info)
    print(len(embeddings))
    print(embeddings[:5])
    return embeddings

# HASHED PASSWORD
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# EXISTING USER CHECK
def existing_user(user_id):
    res = client.get(
    collection_name=COLLECTION_NAME,
    ids=[user_id]
    )   
    if res != []:
        return True
    else:
        return False

# CREATE NEW USER
def create_new_user(user_id):
    tempData = {
    "id": user_id,
    "user_info": "",
    "info_vector": embed_info("")
    }
    res = client.insert(
    collection_name=COLLECTION_NAME,
    data=tempData
    )
    print(res)

# UPDATE USER INFO
def update_user_info(tempData):
    res = collection.upsert(
    data=tempData
    )

def get_user_data_from_database(user_id):
    res = client.get(
    collection_name=COLLECTION_NAME,
    ids=[user_id]
    )
    info = str([item["user_info"] for item in res])
    return info

# CONVERT PLANT DATA TO STRING
def save_plant_data_to_string(plant_data_list):
    """
    Converts a list of plant data dictionaries into a single string.

    Parameters:
    plant_data_list (list): A list of dictionaries, each containing plant data.

    Returns:
    str: A single string containing all plant data.
    """
    # Initialize an empty string to store plant information
    plants_info = ""
    
    # Loop through each dictionary in the list and convert it to a string
    for plant_data in plant_data_list:
        plant_info = ", ".join(f"'{key}': {value}" for key, value in plant_data.items() if value)
        plants_info += "{ " + plant_info + " }\n"
    
    return plants_info

def create_embeddings_from_image_bytes(image_bytes):
    # Initialize the feature extractor and model
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

    # Convert bytes data to a file-like object
    img = Image.open(io.BytesIO(image_bytes))

    # Preprocess the image using the feature extractor
    inputs = extractor(images=img, return_tensors="pt")

    # Forward pass through the model
    outputs = model(**inputs)

    # Extract the embeddings from the output
    logits = outputs.logits
    return logits

def perform_vector_search(embeddings):
    # Convert PyTorch tensor to a list of lists
    vectors_to_search = embeddings.detach().cpu().tolist()

    # If logits is a 2D tensor with a single data point, it will be a nested list after tolist(),
    # but Milvus expects a flat list for a single vector, so take the first element.
    if len(vectors_to_search) == 1:
        vectors_to_search = vectors_to_search[0]

    # Now perform the search with the vector
    res = client.search(
        collection_name='plant_diseases',
        data=[vectors_to_search],  # Wrap the flat list in another list to make it a list of lists
        output_fields=["label"],
        limit=5
    )

    return res

