import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import GooglePaLMEmbedding
from llama_index.llms.palm import PaLM

from langchain.llms import VertexAI
from trulens_eval import TruLlama, Tru, OpenAI


from trulens_eval import Feedback
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
from trulens_eval import TruLlama

import numpy as np
import logging

import os
from dotenv import load_dotenv
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Streamlit page configuration
st.set_page_config(page_title="LLAMAINDEX Query Engine Evaluation", layout="wide")
st.title("RAG - QUERY ENGINE 🌾🌱 ")




# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_PALM_KEY")





# Run predefined evaluation queries
evaluation_queries = [
    "What is the best crop rotation practice for wheat farming?",
    "What is the best crop rotation practice for corn farming?",
    "What is the best crop rotation practice for soybean farming?",
]

openai = OpenAI(api_key=openai_api_key)


fopenai = fOpenAI(client = openai)

# Initialize feedback functions for RAG TRIAD

grounded = Groundedness(groundedness_provider=fopenai)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(TruLlama.select_source_nodes().node.text.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_qa_relevance = (
    Feedback(fopenai.relevance_with_cot_reasons, name = "Answer Relevance")
    .on_input_output()
)

# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(fopenai.qs_relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

feedbacks = [f_groundedness, f_qa_relevance, f_context_relevance]

# Error handling wrapper
def handle_api_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error encountered: {e}")
            st.error(f"An error occurred: {e}")
            return None
    return wrapper
# Data Loading
with st.container():
    st.header("Data Loading")
    progress_bar_data = st.progress(0)
    documents = SimpleDirectoryReader('rag_data').load_data()  # Replace with your data directory
    progress_bar_data.progress(100)
    st.success("Data loaded successfully")

# Query Engine Initialization
with st.container():
    st.header("Query Engine #1 Initialization")
    progress_bar_engine = st.progress(0)
    vertex = PaLM(api_key=google_api_key)
    embed_model = GooglePaLMEmbedding(model_name="models/embedding-gecko-001", api_key=google_api_key)
    service_context = ServiceContext.from_defaults(llm=vertex, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    progress_bar_engine.progress(100)
    st.success("Query engine initialized")

# Evaluation
with st.container():
    st.header("Evaluation")
    progress_bar_eval = st.progress(0)
    tru = Tru()
    tru.reset_database()
    tru_recorder = TruLlama(query_engine, app_id="Query-Engine-1", feedbacks=feedbacks)

    @handle_api_errors
    def execute_query(query):
        with tru_recorder as recording:
            return query_engine.query(query)

    for i, query in enumerate(evaluation_queries, start=1):
        response = execute_query(query)
        if response:
            st.write(f"Executing Query {i}: {query}")
            st.write(f"Response: {response}")
        progress_bar_eval.progress(10 + (i * (90 // len(evaluation_queries))))

    st.success("Evaluation completed")

# Displaying Results
with st.expander("Detailed Evaluation Results"):
    records, feedback = tru.get_records_and_feedback(app_ids=[])
    st.dataframe(records)
    
with st.container():
    st.header("Evaluation")    
    st.dataframe(tru.get_leaderboard(app_ids=[]))
    st.dataframe(feedback)