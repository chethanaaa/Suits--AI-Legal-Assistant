# Import necessary modules
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import RetrievalQA
import logging
import json

# Load API key from config.json
with open("config.json", "r") as file:
    config = json.load(file)

openai_key = config["openai_key"]

# Suppress warnings
logging.getLogger("cryptography").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Load PDF documents
pdf_paths = ["./data/RLTO Summary_2023_EN_FINAL.pdf"]
documents = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["source"] = pdf_path.split("/")[-1]
        doc.metadata["page"] = i + 1
        documents.append(doc)

# Create document embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vector_store = FAISS.from_documents(documents, embeddings)

# Set up language model and RAG chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)

# Function to handle user input intelligently
def handle_user_input(user_input):
    # Define greetings for conversational responses
    greetings = ["hi", "hello", "hey", "hi there", "hello there"]

    # Check if the input is a greeting
    if user_input.lower() in greetings:
        return "Hi there! How can I assist you with Chicago Housing Laws today?"

    # If not a greeting, proceed with legal document summarization
    return summarize_legal_documents(user_input)

def summarize_legal_documents(user_input):
    # Retrieve relevant content
    retrieval_response = qa_chain({"query": user_input})
    
    # Extract relevant pages and texts
    retrieved_texts = [
        f"Document: '{doc.metadata.get('source', 'Unknown Document')}', Page {doc.metadata.get('page', 'N/A')}:\n{doc.page_content}"
        for doc in retrieval_response["source_documents"]
    ]
    retrieved_texts_combined = "\n\n".join(retrieved_texts)

    # Generate summary prompt
    prompt = (
        "Your name is Suits, an empathetic and sensitive AI Legal Assistant specializing in Chicago Housing Laws. "
        "Engage warmly and conversationally. If the user greets you, respond warmly and ask how you can assist. "
        "If the user asks a legal question, summarize the relevant laws or documents in bullet points. "
        "Provide concise responses directly from the retrieved documents with citations (document names, page numbers, and sections).\n\n"
        f"Query: {user_input}\n"
        f"Retrieved content:\n{retrieved_texts_combined}\n\n"
        "Summary or Response:"
    )

    # Generate response using the LLM
    response = llm([HumanMessage(content=prompt)])
    assistant_response = response.content

    # Extract unique citations for the most relevant documents only
    unique_citations = set()
    for doc in retrieval_response["source_documents"]:
        doc_name = doc.metadata.get("source", "Unknown Document")
        page_number = doc.metadata.get("page", "N/A")
        citation = f"<small style='color: gray;'>Source: {doc_name}, Page {page_number}.</small>"
        unique_citations.add(citation)

    # Save only the most relevant retrieved texts and citations
    st.session_state["retrieved_texts"] = [retrieved_texts[0]]  # Show only the first relevant document
    st.session_state["citations"] = list(unique_citations)[:1]  # Show citation for the first relevant document

    return assistant_response


# Initialize Streamlit app
st.title("Suits: Legal Document Summarizer and Conversational Assistant")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "retrieved_texts" not in st.session_state:
    st.session_state["retrieved_texts"] = []

if "citations" not in st.session_state:
    st.session_state["citations"] = []

# User input handling
if user_input := st.chat_input("Enter your query or response here..."):
    st.session_state["chat_history"].append({"role": "User", "content": user_input})
    assistant_response = handle_user_input(user_input)
    st.session_state["chat_history"].append({"role": "Assistant", "content": assistant_response})

# Display chat history
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"].lower()):
        st.markdown(message["content"])

# Show citations if checkbox is enabled
if st.checkbox("Show citations"):
    st.markdown("**Citations:**")
    for citation in st.session_state.get("citations", []):
        st.markdown(citation, unsafe_allow_html=True)

# Show retrieved content if checkbox is enabled
if st.checkbox("Show retrieved content"):
    for doc in st.session_state.get("retrieved_texts", []):
        st.text(doc)

# Feedback handling
if st.button("Provide Feedback"):
    feedback = st.text_area("Enter your feedback here:")
    if feedback:
        st.session_state["chat_history"].append({"role": "User Feedback", "content": feedback})
        st.success("Your feedback has been recorded. Thank you!")

# Disclaimer
st.markdown(
    "<small style='color: gray;'>Disclaimer: This AI Legal Assistant is for informational purposes only and relies solely on the provided documents. It is not a substitute for professional legal advice.</small>",
    unsafe_allow_html=True,
)
