# import ipywidgets as widgets
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.schema import HumanMessage, AIMessage
# from langchain.chains import RetrievalQA
# import streamlit as st
# import logging

# # Suppress specific warnings
# logging.getLogger("cryptography").setLevel(logging.ERROR)
# logging.getLogger("pypdf").setLevel(logging.ERROR)

# # Load the PDF document and initialize embeddings
# # loader = PyPDFLoader("/Users/Chethana/Downloads/RLTO Summary_2023_EN_FINAL.pdf")
# # documents = loader.load()  # Load the PDF content


# # List of PDF file paths
# pdf_paths = [
#     "/Users/chethana/Downloads/RLTO Summary_2023_EN_FINAL.pdf",
#     "/Users/chethana/Downloads/14X-8-802 HEATING SYSTEMS_.pdf",
#     "/Users/chethana/Downloads/Municipal_Code_Chapter_5-14.pdf",
#     "/Users/chethana/Downloads/775 ILCS 5_ Illinois Human Rights Act_.pdf"
# ]

# # Initialize an empty list to store all documents
# all_documents = []

# # Iterate over the list of file paths and load each document
# for pdf_path in pdf_paths:
#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()
#     all_documents.extend(documents)

# # Create document embeddings
# embeddings = OpenAIEmbeddings(openai_api_key="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")
# document_texts = [doc.page_content for doc in documents]  # Extract page content
# vector_store = FAISS.from_texts(document_texts, embeddings)

# # Set up the language model
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

# # Initialize the RAG chain with your document retriever
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
    
#     chain_type="stuff",
#     retriever=vector_store.as_retriever(),
#     return_source_documents=True
# )

# # Function to generate the prompt
# def generate_prompt(user_input):
#     prompt = (
#         "Your name is Suits, an empathetic and concise AI Legal Assistant specializing in Chicago Housing Laws. "
#         "Ask relevant clarifying questions to understand the user's situation better. Gather specific details from the user before giving any final advice. "
#         "Provide brief, clear responses directly from the retrieved documents. Include specific citations with document names and page numbers.\n\n"
#     )
#     for role, content in st.session_state["chat_history"]:
#         prompt += f"{role}: {content}\n"
#     prompt += f"User: {user_input}\nAssistant: "
#     return prompt

# # Function to handle the assistant's response
# def interactive_legal_assistant(user_input):
#     prompt = generate_prompt(user_input)
#     retrieval_response = qa_chain({"query": user_input})
    
#     # Prepare retrieved texts for display
#     retrieved_texts = [
#         f"Document: '{doc.metadata.get('document_name', 'Unknown Document')}', Page {doc.metadata.get('page_number', 'N/A')}: {doc.page_content}"
#         for doc in retrieval_response["source_documents"]
#     ]
    
#     # Add additional information to the prompt
#     prompt += (
#         "\n\nUsing the information from the retrieved documents, provide brief, clear responses with specific citations. "
#         "ASK ONE LINER OR TWO LINER CLARIFYING QUESTIONS TO GET AS MUCH CONTEXT AS POSSIBLE before making a decision. "
#         "DON'T BE REPETITIVE IN THE WORDINGS, BE EMOTIONAL BE HUMAN SOUNDING AND DON'T ASK BIG QUESTIONS"
#         "Keep responses concise and to the point, directly addressing the user’s query.\n"
#         "When you have enough information after asking SHORT clarifying questions, make a well-reasoned decision based on the laws and provide a clear response citing laws and references and citations with page numbers. "
#         f"Retrieved document content:\n{retrieved_texts}\n\nAssistant:"
#     )
    
#     # Get the assistant's response
#     response = llm([HumanMessage(content=prompt)])
#     return response.content

# # Initialize Streamlit app
# st.title("Suits")

# # Initialize chat history in session state if it doesn't exist
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []

# try:
#     chat_history = st.session_state["chat_history"]
# except KeyError:
#     st.session_state["chat_history"] = []
#     chat_history = st.session_state["chat_history"]

# # User input and assistant response handling
# if user_input := st.chat_input("Enter your question or response here..."):
#     # Add user input to chat history
#     st.session_state["chat_history"].append(("User", user_input))
    
#     # Get and add assistant response to chat history
#     assistant_response = interactive_legal_assistant(user_input)
#     st.session_state["chat_history"].append(("Assistant", assistant_response))

# # Display all messages in chat history
# for role, content in st.session_state["chat_history"]:
#     with st.chat_message(role.lower()):
#         st.markdown(content)




# Import necessary modules
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import RetrievalQA
import logging
from chains import create_decision_chain, create_rules_chain
import json
from utils import *

# Load the kb data
with open('./kb2.json', 'r') as file:
    kb = json.load(file)

# Suppress warnings
logging.getLogger("cryptography").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)

decision_chain = create_decision_chain(openai_key)
rules_chain = create_rules_chain(openai_key)
# Load PDF documents
pdf_paths = [
    "./RLTO Summary_2023_EN_FINAL.pdf",
    "./14X-8-802 HEATING SYSTEMS_.pdf",
    "./Municipal_Code_Chapter_5-14.pdf",
    "./775 ILCS 5_ Illinois Human Rights Act_.pdf"
]
documents = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())

# Create document embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
document_texts = [doc.page_content for doc in documents]
vector_store = FAISS.from_texts(document_texts, embeddings)

# Set up language model and RAG chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Function to generate the prompt
def generate_prompt(user_input):
    prompt = (
        "Your name is Suits, an empathetic and sensitive AI Legal Assistant specializing in Chicago Housing Laws. "
        "Engage warmly and conversationally. If the user greets you, respond warmly and ask how you can assist. "
        "Ask short, empathetic follow-up questions to understand the user's situation better. "
        "Provide brief, concise responses directly from the retrieved documents with clear citations (document names, page numbers, and sections). "
        "If insufficient context is available, ask clarifying questions instead of guessing or giving incomplete advice. "
        "Do not suggest consulting a legal professional. Gather as much context as possible by asking questions."
        "Avoid repeating phrases unnecessarily. Use a friendly and supportive tone.\n\n"
    )
    for role, content in st.session_state["chat_history"]:
        prompt += f"{role}: {content}\n"
    prompt += f"User: {user_input}\nAssistant: "
    return prompt

# Interactive assistant function
def interactive_legal_assistant(user_input):
    prompt = generate_prompt(user_input)
    retrieval_response = qa_chain({"query": user_input})

    # Prepare retrieved documents for citation
    retrieved_texts = [
        f"Document: '{doc.metadata.get('document_name', 'Unknown Document')}', Page {doc.metadata.get('page_number', 'N/A')}: {doc.page_content}"
        for doc in retrieval_response["source_documents"]
    ]

    # Update prompt to enforce citations
    prompt += (
        "\n\nUsing the retrieved documents, respond empathetically and conversationally. "
        "Ask additional clarifying questions if context is insufficient. "
        "When giving advice, cite relevant documents with section names and page numbers. Avoid guessing. "
        f"Retrieved document content:\n{retrieved_texts}\n\nAssistant:"
    )

    # Generate response
    response = llm([HumanMessage(content=prompt)])
    assistant_response = response.content

    # Evaluate decision readiness
    conversation = "\n".join([f"{role}: {content}" for role, content in st.session_state["chat_history"]])
    decision_result = decision_chain.invoke({"conversation": conversation})
    # st.session_state["chat_history"].append(("Assistant", assistant_response))
    # Check for context completeness
    # if "decision" in assistant_response.lower() or "suggestion" in assistant_response.lower():
    #     context_complete = True
    print("Decision: YES OR NO: ")
    print(decision_result.decision_ready)

    if decision_result.decision_ready == "yes":
        # Determine the applicable rule
        rule_result = rules_chain.invoke({"case_details": conversation})
        print(f"RULE RESULT: {rule_result.selected_rule}")
        action = get_action_by_rule(kb, rule_result.selected_rule)
        compensation = get_compensation_details(kb, rule_result.selected_rule)
        st.session_state["chat_history"].append(("System", f"Applicable Rule: {rule_result.selected_rule}, Action: {action}, Compensation: {compensation}"))
        return assistant_response, rule_result.selected_rule

    # Append response to chat history
    #st.session_state["chat_history"].append(("Assistant", assistant_response))
    return assistant_response, None

# Initialize Streamlit app
st.title("Suits: AI Legal Assistant for Chicago Housing Laws")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input handling
if user_input := st.chat_input("Enter your query or response here..."):
    st.session_state["chat_history"].append(("User", user_input))
    assistant_response, rule = interactive_legal_assistant(user_input)
    st.session_state["chat_history"].append(("Assistant", assistant_response))
    # if rule != None:
    #     st.session_state["chat_history"].append(("System", f"Applicable Rule: {rule}"))


if st.button("Provide Feedback"):
    feedback = st.text_area("Enter your feedback here:")
    if feedback:
        st.session_state["chat_history"].append(("User Feedback", feedback))
        st.success("Your feedback has been recorded. Thank you!")


# Display chat history
for role, content in st.session_state["chat_history"]:
    with st.chat_message(role.lower()):
        st.markdown(content)

# Disclaimer
st.markdown(
    "<small style='color: gray;'>Disclaimer: This AI Legal Assistant is for informational purposes only and relies solely on the provided documents. It is not a substitute for professional legal advice.</small>",
    unsafe_allow_html=True
)
