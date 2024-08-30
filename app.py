# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain.vectorstores import Pinecone
# from pinecone import Pinecone, ServerlessSpec

# from langchain import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.llms import CTransformers
# from dotenv import load_dotenv
# from src.prompt import *
# import os

# app = Flask(__name__)


# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# embeddings = download_hugging_face_embeddings()

# pc = Pinecone(api_key=PINECONE_API_KEY,
#     serverless_spec=ServerlessSpec(
#         cloud='aws',
#         region="us-east-1"  # Set your desired region here
#         ) 
# )

# def text_split(extracted_data):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     text_chunks = text_splitter.split_documents(extracted_data)

#     return text_chunks
# text_chunks = text_split(extracted_data)
# print("length of my chunks:", len(text_chunks))

# index_name="medical-chatbot"
# index = pc.Index(index_name)  
# for i, t in zip(range(len(text_chunks)), text_chunks):
#    query_result = embeddings.embed_query(t.page_content)
#    index.upsert(
#    vectors=[
#         {
#             "id": str(i),  # Convert i to a string
#             "values": query_result, 
#             "metadata": {"text":str(text_chunks[i].page_content)} # meta data as dic
#         }
#     ],
#     namespace="real" 
# )
   


# PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# chain_type_kwargs={"prompt": PROMPT}

# llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})


# index_name = 'medical-chatbot'
# index = pc.Index(index_name)  # Reconnect to your Pinecone index

# def retrieve_documents(query, top_k=2):
#     # Embed the user's query
#     query_embedding = embeddings.embed_query(query)
    
#     # Perform similarity search in the Pinecone index
#     search_result = index.query(
#         vector=query_embedding,
#         top_k=top_k,
#         include_metadata=True,
#         namespace="real"
#     )
    
#     # Extract the retrieved documents (metadata["text"])
#     documents = [match['metadata']['text'] for match in search_result['matches']]
#     return documents

# def generate_response(llm, documents, query):
#     # Combine the retrieved documents with the query to generate a response
#     context = "\n\n".join(documents)
    
#     # Generate the response using the LLM
#     full_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
#     response = llm(full_prompt)
    
#     return response


# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     result=qa({"query": input})
#     print("Response : ", result["result"])
#     return str(result["result"])


# if __name__ == '__main__':
#     app.run(host="0.0.0.0",port=8080,debug= True)

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
# from langchain.vectorstores import Pinecone
# import pinecone
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
    )
)

index_name = "medical-chatbot"

# docsearch=Pinecone.from_existing_index(index_name, embeddings)


#Retrieve documents
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


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
# Generate response using LLM
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens': 512, 'temperature': 0.8})

# def generate_response(llm, documents, query):
#     context = "\n\n".join(documents)
#     full_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
#     response = llm(full_prompt)
#     return response


@app.route("/")
def index():
    return render_template('chat.html')

#Flask route to handle user queries
@app.route('/query', methods=['POST'])
def query():
    user_input = request.json.get('query')
    retrieved_docs = retrieve_documents(user_input, top_k=2)
    final_response = generate_response(llm, retrieved_docs, user_input)
    return jsonify({"response": final_response})

if __name__ == '__main__':
    app.run(debug=True)
