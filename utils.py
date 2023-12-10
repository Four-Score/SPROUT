from llama_index.embeddings import HuggingFaceEmbedding
import hashlib
import json, time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient
import os
import json
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env
CLUSTER_ENDPOINT= os.getenv("CLUSTER_ENDPOINT") # Set your cluster endpoint
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
def save_plants_data(plants_data):
    # Save plant data to a file
    with open('plant_data.json', 'w') as file:
        json.dump(plants_data, file)
def get_plants_data():
    # Retrieve plant data from a file
    try:
        with open('plant_data.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []  # Return an empty list if the file does not exist

