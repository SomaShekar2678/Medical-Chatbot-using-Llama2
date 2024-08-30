from flask import Flask, render_template, jsonify, request
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from src.helper import download_hugging_face_embeddings, generate_response
from langchain.chains import RetrievalQA
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_hugging_face_embeddings()

# Pinecone initialization
pc = Pinecone(api_key=PINECONE_API_KEY,
              serverless_spec=ServerlessSpec(
                  cloud='aws',
                  region="us-east-1"
              ))

index_name = "medical-chatbot"

# Retrieve documents
def retrieve_documents(query, top_k=2):
    query_embedding = embeddings.embed_query(query)
    search_result = pc.Index(index_name).query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="real"
    )
    documents = [match['metadata']['text'] for match in search_result['matches']]
    return documents

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Generate response using LLM
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.8})

@app.route("/")
def index():
    return render_template('chat.html')

# Flask route to handle user queries
@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form["msg"]
    user_input = msg  # Renamed variable to avoid conflict
    print(user_input)  # Correct print statement
    retrieved_docs = retrieve_documents(user_input, top_k=2)
    final_response = generate_response(llm, retrieved_docs, user_input)
    print("Response: ", final_response)
    return str(final_response)

if __name__ == '__main__':
    app.run(debug=True)
