import streamlit as st
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import json

with open("config.json", "r") as file:
    config = json.load(file)

openai_key = config["openai_key"]

client = OpenAI(api_key=openai_key)

st.set_page_config(layout="wide")

# Load and process documents for retrieval
@st.cache_resource
def prepare_retrieval_system():
    loader = PyPDFLoader("/Users/saichakradhar/Downloads/suits/RLTO Summary_2023_EN_FINAL.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Initialize the FAISS vector store
vector_store = prepare_retrieval_system()

# Define personas for judges and lawyers
judges_personas = [
    {
        "name": "Legal Expert",
        "prompt": (
            "You are a Legal Expert. Respond from the perspective of someone who strictly interprets the law. "
            "Using the following context, provide a judgment on the case: \"{}\""
        )
    },
    {
        "name": "Pragmatist",
        "prompt": (
            "You are a Pragmatist. Respond from the perspective of someone focused on real-world outcomes and consequences. "
            "Using the following context, provide your judgment on the case: \"{}\""
        )
    },
    {
        "name": "Society's Representative",
        "prompt": (
            "You are Society's Representative. Respond from the perspective of someone considering societal impact and precedent. "
            "Using the following context, provide your view on the case: \"{}\""
        )
    }
]

lawyer_personas = [
    {
        "name": "Aggressive Defense Lawyer",
        "prompt": (
            "You are an Aggressive Defense Lawyer. Respond from the perspective of a lawyer who passionately defends the accused "
            "and seeks to exploit every legal technicality. Using the following context, provide your argument: \"{}\""
        )
    },
    {
        "name": "Compassionate Defense Lawyer",
        "prompt": (
            "You are a Compassionate Defense Lawyer. Respond from the perspective of a lawyer who emphasizes the humanity of the "
            "accused and seeks leniency based on their circumstances. Using the following context, provide your argument: \"{}\""
        )
    },
    {
        "name": "Prosecution Lawyer",
        "prompt": (
            "You are a Prosecution Lawyer. Respond from the perspective of someone who seeks justice for the victim and society, "
            "focusing on proving the accused's guilt. Using the following context, provide your argument: \"{}\""
        )
    }
]

# Function to generate clarifying questions
def generate_clarifying_questions(input_text):
    system_prompt = (
        "You are a helpful assistant tasked with understanding a legal case. The user has provided the following details:\n"
        f"\"{input_text}\"\n"
        "Identify up to 5 simple clarifying questions necessary to analyze this case comprehensively. "
        "These questions should help gather critical missing details."
    )
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return completion.choices[0].message.content.strip()


# Streamlit UI for user input
st.title("Persona Simulator")

# Step 1: Get initial input
case_details = st.text_area("Enter the case details:", placeholder="Describe the case...")

if case_details:
    # Step 2: Generate clarifying questions if input is insufficient
    if len(case_details.split()) < 2000:  # Arbitrary word count threshold
        st.warning("The initial input seems insufficient. Let's gather more details.")
        clarifying_questions = generate_clarifying_questions(case_details)
        st.write("Here are some questions to help provide more context:")
        st.text(clarifying_questions)

        # Allow user to answer the clarifying questions
        additional_details = st.text_area(
            "Answer the above questions to provide more information:",
            placeholder="Provide additional details here..."
        )
        # Check if additional details are provided
        if not additional_details:
            st.stop()  # Stop execution until answers are provided
        else:
            case_details += " " + additional_details


    # Step 3: Retrieve relevant context from the vector store
    docs = vector_store.similarity_search(case_details, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create two wide columns
    col1, col2 = st.columns([5, 5])  # Adjusted to elongate the columns

    # Judges Block in the left column
    with col1:
        st.header("Judges' Opinions")
        for judge in judges_personas:
            # Format the judge's prompt with the case details and context
            prompt = judge["prompt"].format(f"Case: {case_details}\nContext: {context}")

            # Call the OpenAI API
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt}
                ]
            )

            # Display the output in an expandable box
            with st.expander(judge["name"]):
                st.write(completion.choices[0].message.content.strip())

    # Lawyers Block in the right column
    with col2:
        st.header("Lawyers' Arguments")
        for lawyer in lawyer_personas:
            # Format the lawyer's prompt with the case details and context
            prompt = lawyer["prompt"].format(f"Case: {case_details}\nContext: {context}")

            # Call the OpenAI API
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt}
                ]
            )

            # Display the output in an expandable box
            with st.expander(lawyer["name"]):
                st.write(completion.choices[0].message.content.strip())


# Disclaimer
st.markdown(
    "<small style='color: gray;'>Disclaimer: This AI Legal Assistant is for informational purposes only and relies solely on the provided documents. It is not a substitute for professional legal advice.</small>",
    unsafe_allow_html=True
)