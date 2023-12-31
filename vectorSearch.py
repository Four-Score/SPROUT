#this is the initial approach we used to implement vector search but due to google cloud credits being all used up, we had to move to a different approach.

import streamlit as st

from google.cloud import aiplatform_v1beta1
from google.oauth2 import service_account
from vertexembed import encode_images_to_embeddings
import os
import json
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env
import toml


def findneighbor_sample(embeddings):
  # The AI Platform services require regional API endpoints.
  scopes = ["https://www.googleapis.com/auth/cloud-platform"]

  # create a service account with `Vertex AI User` role granted in IAM page.
  # download the service account key https://developers.google.com/identity/protocols/oauth2/service-account#authorizingrequests
# Access the credentials
 # config = st.secrets["google_credentials"]

  # Construct a credentials object from the dictionary
  #credentials = service_account.Credentials.from_service_account_info(config)
  #client_options = {
     # "api_endpoint": "231746582.us-central1-145895176016.vdb.vertexai.goog"
#  }

 # vertex_ai_client = aiplatform_v1beta1.MatchServiceClient(
     # credentials=credentials,
    #  client_options=client_options,
 # )

  #request = aiplatform_v1beta1.FindNeighborsRequest(
  #    index_endpoint="projects/145895176016/locations/us-central1/indexEndpoints/6932803443174146048",
      #deployed_index_id="multimodal_plants_deployed_1701961999298",
#  )

  dp1 = aiplatform_v1beta1.IndexDatapoint(
      datapoint_id="0",
      feature_vector=embeddings,
  )
  query = aiplatform_v1beta1.FindNeighborsRequest.Query(
      datapoint=dp1,
  )
  request.queries.append(query)

  response = vertex_ai_client.find_neighbors(request)

  return response
