from dotenv import load_dotenv
import os
import streamlit as st
from google.cloud import aiplatform
from google.oauth2 import service_account
import base64
from google.cloud.aiplatform.gapic.schema import predict as schema_predict
from PIL import Image
import io
import json

# Load environment variables from .env file
load_dotenv()
import toml
#config = st.secrets["google_credentials"]
# Construct a credentials object from the dictionary

#creds = service_account.Credentials.from_service_account_info(config)
# Parse the service account credentials from the environment variable


# Define the function to make a prediction
def make_prediction(image_bytes):
    client_options = {"api_endpoint": f"{os.getenv('LOCATION')}-aiplatform.googleapis.com"}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options, credentials=creds)
    
    encoded_content = base64.b64encode(image_bytes).decode("utf-8")
    instance = schema_predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    parameters = schema_predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5,
        max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=os.getenv("PROJECT_ID_CODE"),
        location=os.getenv("LOCATION"),
        endpoint=os.getenv("ENDPOINT_ID")
    )
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)

    # Parse the response to make it human-readable
    predictions = []
    for prediction in response.predictions:
        predictions.append(dict(prediction))

    return predictions
