import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import requests

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Step 1: Load and split PDF
loader = PyPDFLoader("1mb.pdf")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Step 2: Create vector DB
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding)

# Step 3: Define RAG function
def ask_rag(query):
    rel_docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in rel_docs])

    prompt = f"""Answer the question based on the context below.
    
    Context:
    {context}

    Question: {query}

    Answer:"""

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
    )

    return res.json()['choices'][0]['message']['content'].strip()

# Step 4: Use chatbot in loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    answer = ask_rag(user_input)
    print("Bot:", answer)
