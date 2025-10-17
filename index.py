from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from  langchain_qdrant.vectorstores import Qdrant 

load_dotenv()
file_path = "./example_data/layout-parser-paper.pdf"
loader = PyPDFLoader(file_path)
##load documents from pdf
docs = loader.load()

#convert into chunks of text
#different types of text splitters are available
#e.g. RecursiveCharacterTextSplitter, CharacterTextSplitter
#definition of each splitter is available in the documentation

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400,
)

chunks = text_splitter.split_documents(docs)

#now you got chunks of text from the  pdf
#now convert using vector model into vectoer embeddings
#and store in vector database
#e.g. FAISS, Pinecone, Weaviate, etc.

embedding_model=OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vectorstore=Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

