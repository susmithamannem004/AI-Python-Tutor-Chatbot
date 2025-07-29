import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
# Import WebBaseLoader for fetching content from URLs
from langchain.document_loaders import WebBaseLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Configuration ---
# You need a Google API key.
# Go to https://aistudio.google.com/ and create one.
# IMPORTANT: Store this in Streamlit secrets or an environment variable, not directly in the code.
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- Knowledge Base and RAG Pipeline Functions ---
def create_vector_store():
    """
    Loads content from specified URLs, splits it, and creates a Chroma vector store.
    This replaces loading from local files.
    """
    # Define a list of URLs to scrape for your knowledge base
    # You can add more relevant URLs here for Python and AI topics
    urls = [
        "https://www.python.org/doc/essays/blurb/", # Official Python intro
        "https://www.geeksforgeeks.org/python-programming-language/", # GeeksforGeeks Python overview
        "https://www.ibm.com/topics/artificial-intelligence", # IBM AI overview
        "https://www.ibm.com/topics/machine-learning", # IBM ML overview
        "https://www.ibm.com/topics/natural-language-processing", # IBM NLP overview
        "https://www.analyticsvidhya.com/blog/2021/07/build-a-simple-chatbot-using-python-and-nltk/", # Chatbot basics
        "https://www.datacamp.com/community/tutorials/machine-learning-algorithms-python", # ML Algorithms
        "https://huggingface.co/docs/transformers/index" # Hugging Face Transformers documentation intro
    ]

    # 1. Load documents from URLs using WebBaseLoader
    # This will fetch the content from each URL
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    # 2. Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Adjust as needed
        chunk_overlap=200 # Adjust as needed
    )
    texts = text_splitter.split_documents(docs)
    
    # 3. Create embeddings for each chunk
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 4. Create and persist the vector store
    # The persist_directory will store the embeddings locally after the first run
    vector_store = FAISS.from_documents(
        documents=texts,
        embedding=embeddings,
        
    )
    return vector_store

def get_rag_chain(vector_store):
    """Initializes the Retrieval-Augmented Generation (RAG) chain."""
    
    # Initialize the Gemini model for generation
    # Using gemini-1.5-flash as it's often more available and efficient
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash") 
    
    # Create a retriever from your vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks
    
    # Define a custom prompt for the chatbot
    template = """
    You are 'Py-AI Tutor', a highly knowledgeable and patient assistant specializing in Python programming and Artificial Intelligence concepts. Your primary goal is to provide clear, concise, and accurate answers based *only* on the information provided in the 'Context' section below.

    If the 'Context' does not contain the answer to the user's question, or if the information is insufficient, please respond politely by stating: "I apologize, but I don't have enough specific information in my current knowledge base to answer that question. Can I help you with something else related to Python or AI?"

    Always strive to be helpful and encouraging. If a code example is relevant and present in the context, please include it.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' puts all retrieved docs into the prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- Streamlit UI and Logic ---
st.set_page_config(page_title="AI & Python Tutor", page_icon="🤖")

st.title("🤖 AI & Python Tutor Chatbot")
st.caption("A helpful assistant to answer your questions on Python and Artificial Intelligence.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Cache the RAG chain to avoid re-initializing on every rerun
@st.cache_resource
def get_qa_chain():
    vector_store = create_vector_store()
    return get_rag_chain(vector_store)

qa_chain = get_qa_chain()

# Main chat input logic
if prompt := st.chat_input("Ask me about Python or AI..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get the response from the RAG chain
        with st.spinner("Thinking..."):
            response = qa_chain({"query": prompt})
            answer = response["result"]
            
            # This simulates a streamed response
            for chunk in answer.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
