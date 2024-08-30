from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def load_pdf(data):

    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

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

def generate_response(llm, documents, query):
    # Combine the retrieved documents with the query to generate a response
    context = "\n\n".join(documents)
    
    # Generate the response using the LLM
    full_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm(full_prompt)
    
    return response
