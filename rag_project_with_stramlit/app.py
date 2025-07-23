import os
import shutil
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
VECTORSTORE_PATH = "faiss_index_streamlit"
TEMP_UPLOAD_DIR = "temp_uploads_streamlit"

# --- INITIALIZE MODELS (Cached for performance) ---
@st.cache_resource
def load_llm():
    """Load the Groq Chat LLM."""
    return ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

@st.cache_resource
def load_embedding_model():
    """Load the embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = load_llm()
embedding_model = load_embedding_model()

# --- CORE FUNCTIONS ---
def process_document(uploaded_file):
    """
    Processes the uploaded document and creates a vector store.
    """
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    
    # Save the uploaded file temporarily
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_file_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_file_path)
    else:
        st.error("Unsupported file type.")
        return False
        
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create and save the vector store
    try:
        # Clean up old vector store if it exists
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
            
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(VECTORSTORE_PATH)
        st.session_state.document_processed = True
        return True
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return False
    finally:
        # Clean up the temp directory
        shutil.rmtree(TEMP_UPLOAD_DIR)

def get_answer(question):
    """
    Handles the entire RAG process for a given question.
    This version is simplified to be more robust.
    """
    if not os.path.exists(VECTORSTORE_PATH):
        return "No document has been processed yet. Please upload a document first."

    vector_store = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)

    # Always retrieve context from the document
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Create a single, robust prompt
    template = """You are a helpful assistant. Answer the user's question based on the context provided below.
If the context does not contain the answer, state that you could not find the answer in the document.

Context:
{context}

Question:
{question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm
    
    answer = chain.invoke({"context": context, "question": question})
    
    return answer.content


# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Document Q&A", layout="wide")

st.title("📄 AI Document Q&A with Groq & Llama 3")
st.markdown("Upload a document (PDF or DOCX), and ask questions about its content. The AI will now reliably use the document content to answer.")

# Initialize session state
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Upload Your Document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document... This may take a moment."):
                success = process_document(uploaded_file)
                if success:
                    st.success("Document processed successfully!")
                    st.session_state.messages = [] # Clear chat on new doc
                else:
                    st.error("Failed to process the document.")
        else:
            st.warning("Please upload a document first.")

# --- Main Chat Interface ---
st.header("2. Ask Questions")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.document_processed:
        st.warning("Please upload and process a document before asking questions.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.spinner("Thinking..."):
            answer = get_answer(prompt)
            
            # Add assistant response to chat history
            response_message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(response_message)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
