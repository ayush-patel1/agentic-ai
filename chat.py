from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores.qdrant import Qdrant
from openai from OpenAI
load_dotenv()

openai_client=OpenAI()  

embedding_model=OpenAIEmbeddings(
    model="text-embedding-3-large"  
)

vector_db=Qdrant.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

#take user query and convert into embedding

user_query=input("Enter your query:")

search_results=vector_db.similarity_search(
    query=user_query,
)

context="\n\n\n".join(
    [doc.page_content for doc in search_results]
)

SYSTEM_PROMPT=f"""
You are a helpful AI assistant. Use the following context to answer the user query.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
you should only ans the user based on the following context and navigate the user to open the right page number to know more.
Context: {context}
"""

openai_client.chat.completions.create(
    model="gpt-4o-mini",                        
    messages=[
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        },
        {
            "role":"user",
            "content":user_query
        }
    ],
    temperature=0.2
)