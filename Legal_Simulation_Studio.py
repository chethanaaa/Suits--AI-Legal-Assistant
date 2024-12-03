import streamlit as st
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import RetrievalQA
import logging
import json

# Suppress specific warnings
logging.getLogger("cryptography").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)

with open("config.json", "r") as file:
    config = json.load(file)

openai_key = config["openai_key"]


# Load PDF documents
pdf_paths = [
    "/Users/chethana/Downloads/RLTO Summary_2023_EN_FINAL.pdf",
    "/Users/chethana/Downloads/14X-8-802 HEATING SYSTEMS_.pdf",
    "/Users/chethana/Downloads/Municipal_Code_Chapter_5-14.pdf",
    "/Users/chethana/Downloads/775 ILCS 5_ Illinois Human Rights Act_.pdf"
]



# Set page configuration
st.set_page_config(page_title="Legal Simulation Studio", layout="centered")

# Title and description
st.title("🎭 Legal Simulation Studio")
st.markdown(
    "Simulate a court session between opposing sides. Hone your arguments and build strategies based on Chicago Housing Laws."
)

# Load and preprocess documents
documents = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["document_name"] = pdf_path.split("/")[-1]
        doc.metadata["page_number"] = doc.metadata.get("page_number", "N/A")
    documents.extend(docs)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vector_store = FAISS.from_documents(documents, embeddings)

# Initialize QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key),
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)

# Session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "position" not in st.session_state:
    st.session_state["position"] = None

# Position selection
st.markdown("### Support a Position")
position_options = ["Support", "Oppose"]
position = st.radio(
    "Select your position:",
    position_options,
    key="position_selector",
    help="Choose whether to argue in support of or against the case."
)

if position:
    st.session_state["position"] = position

# Case details
st.markdown("### Enter Case Details")
case_details = st.text_area("Provide case details for the simulation:")

# Map document names to URLs
document_urls = {
    "RLTO Summary_2023_EN_FINAL.pdf": "https://www.chicago.gov/content/dam/city/depts/doh/RLTO/RLTO%20Summary_2023_EN_FINAL.pdf",
    "14X-8-802 HEATING SYSTEMS_.pdf": "https://codelibrary.amlegal.com/codes/chicago/latest/chicago_il/0-0-0-2676217",
    "Municipal_Code_Chapter_5-14.pdf": "https://www.chicago.gov/content/dam/city/depts/doh/general/Municipal_Code_Chapter_5-14.pdf",
    "775 ILCS 5_ Illinois Human Rights Act_.pdf": "https://www.ilga.gov/legislation/ilcs/ilcs5.asp?ActID=2266",
}

# Simulation chat
if case_details.strip():
    user_input = st.chat_input("Enter your argument:")
    if user_input:
        st.session_state["chat_history"].append({"role": "User", "content": user_input})

        # Generate counterargument
        prompt = (
    f"You are an AI legal assistant arguing in the position: {st.session_state['position']}.\n"
    "Your task is to engage in a simulated courtroom debate over the given case details. "
    "Craft your responses as a lawyer passionately defending your stance (either 'Support' or 'Oppose'):\n\n"
    "Guidelines:\n"
    "1. Maintain your assigned position throughout the conversation. Do not argue for the opposing stance.\n"
    "2. Base your arguments on the case details and the user's input. Build logical points to either support or oppose the case.\n"
    "3. Acknowledge the user's input empathetically and conversationally while reinforcing your position.\n"
    "4. Provide concise arguments in bullet points. Avoid verbose explanations or unnecessary repetition.\n"
    "5. Include citations with document names as hyperlinks where appropriate but do not repeat the same source multiple times.\n\n"
    f"Case details: {case_details}\nUser argument: {user_input}\nYour position: {st.session_state['position']}\nYour response:"
)

        response = qa_chain({"query": prompt})
        counterargument = response["result"]

        # Extract citations and ensure uniqueness with hyperlinks
        unique_citations = set()
        for doc in response["source_documents"]:
            doc_name = doc.metadata.get("document_name", "Unknown Document")
            doc_url = document_urls.get(doc_name, "#")  # Fallback to "#" if URL not found
            citation = f"<small style='color: gray;'>Source: <a href='{doc_url}' target='_blank'>{doc_name}</a></small>"
            unique_citations.add(citation)

        # Save response to chat history
        st.session_state["chat_history"].append(
            {
                "role": "Assistant",
                "content": counterargument,
                "citations": list(unique_citations),
            }
        )

# Display chat
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"].lower()):
        st.markdown(message["content"])
        if "citations" in message:
            for citation in message["citations"]:
                st.markdown(citation, unsafe_allow_html=True)
