import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generator, List, Optional
from google.oauth2 import service_account
from dotenv import load_dotenv

import base64
import typing
from tqdm.auto import tqdm

from google.cloud import aiplatform, aiplatform_v1beta1
from google.protobuf import struct_pb2

import os
import json
import vertexai
from google.oauth2 import service_account

# Load environment variables from .env file
load_dotenv()
import toml
# Access the credentials
config = st.secrets["google_credentials"]
# Construct a credentials object from the dictionary

creds = service_account.Credentials.from_service_account_info(config)


client_options = {"api_endpoint": f"{os.getenv('LOCATION')}-aiplatform.googleapis.com"}
client = aiplatform.gapic.PredictionServiceClient(client_options=client_options, credentials=creds)

class EmbeddingResponse(typing.NamedTuple):
    image_embedding: typing.Sequence[float]
    label: str

def load_image_bytes(image_uri: str) -> bytes:
    """Load image bytes from a remote or local URI."""
    image_bytes = None
    if image_uri.startswith("http://") or image_uri.startswith("https://"):
        response = requests.get(image_uri, stream=True)
        if response.status_code == 200:
            image_bytes = response.content
    else:
        image_bytes = open(image_uri, "rb").read()
    return image_bytes


class EmbeddingPredictionClient:
    """Wrapper around Prediction Service Client."""

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        api_regional_endpoint: str = f"{os.getenv('LOCATION')}-aiplatform.googleapis.com",
    ):
        client_options = {"api_endpoint": api_regional_endpoint}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options, credentials=creds
        )
        self.location = location
        self.project = project

    def get_embedding(self, image_bytes: bytes, label: str):
        if not image_bytes:
            raise ValueError("Image bytes must be provided.")

        instance = struct_pb2.Struct()
        encoded_content = base64.b64encode(image_bytes).decode("utf-8")
        image_struct = instance.fields["image"].struct_value
        image_struct.fields["bytesBase64Encoded"].string_value = encoded_content

        instances = [instance]
        endpoint = (
            f"projects/{self.project}/locations/{self.location}"
            "/publishers/google/models/multimodalembedding@001"
        )
        response = self.client.predict(endpoint=endpoint, instances=instances)

        image_embedding = None
        if image_bytes:
            image_emb_value = response.predictions[0]["imageEmbedding"]
            image_embedding = [v for v in image_emb_value]

        # Return the embedding and the label
        return {
            'image_embedding': image_embedding,
            'label': label
        }
    
def generate_batches(
    inputs: List[str], batch_size: int
) -> Generator[List[str], None, None]:
    """
    Generator function that takes a list of strings and a batch size, and yields batches of the specified size.
    """

    for i in range(0, len(inputs), batch_size):
        yield inputs[i : i + batch_size]


API_IMAGES_PER_SECOND = 2


def encode_to_embeddings_chunked(
    process_function: Callable[[List[dict]], List[dict]],
    items: List[str],
    batch_size: int = 1,
) -> List[Optional[List[float]]]:
    """
    Function that encodes a list of strings into embeddings using a process function.
    It takes a list of strings and returns a list of optional lists of floats.
    The data is processed in chunks to prevent out-of-memory errors.
    """

    embeddings_with_labels_list: List[dict] = []

    # Prepare the batches using a generator
    batches = generate_batches(items, batch_size)

    seconds_per_job = batch_size / API_IMAGES_PER_SECOND

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(batches, total=len(items) // batch_size, position=0):
            futures.append(executor.submit(process_function, batch))
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_with_labels_list.extend(future.result())
    return embeddings_with_labels_list

import copy
from typing import List, Optional

import numpy as np
import requests
from tenacity import retry, stop_after_attempt

client = EmbeddingPredictionClient(os.getenv('PROJECT_ID'))


# Use a retry handler in case of failure
@retry(reraise=True, stop=stop_after_attempt(3))
@retry(reraise=True, stop=stop_after_attempt(3))
def encode_images_to_embeddings_with_retry(image_data: List[dict]) -> List[dict]:
    # Ensure that each item in the list contains only one image data dictionary
    assert len(image_data) == 1

    image_uri = image_data[0]['image_path']
    label = image_data[0]['label']

    try:
        embedding_response = client.get_embedding(image_file=image_uri, label=label)
        return [embedding_response._asdict()]  # Convert the namedtuple to a dictionary
    except Exception as ex:
        print(ex)
        raise RuntimeError("Error getting embedding for image: {}".format(image_uri))

def encode_images_to_embeddings(image_bytes: bytes, label: str) -> dict:
    """
    Function that encodes image bytes into embeddings using the process function.
    It takes image bytes and a label, and returns a dictionary with the embedding and the label.
    """
    client = EmbeddingPredictionClient('145895176016')  # Assuming you have a way to instantiate this

    try:
        embedding_response = client.get_embedding(image_bytes=image_bytes, label=label)
        return embedding_response  # It should already be a dictionary
    except Exception as ex:
        print(ex)
        # Return None for the embedding in case of an error, but keep the label
        return {'image_embedding': None, 'label': label}



"""image_embeddings_with_labels = encode_to_embeddings_chunked(
    process_function=encode_images_to_embeddings, items=image_data_filtered
)

# Keep only non-None embeddings and their corresponding labels
valid_embeddings_with_labels = [
    data for data in image_embeddings_with_labels if data['image_embedding'] is not None
]

print(f"Processed {len(valid_embeddings_with_labels)} embeddings successfully")
"""
