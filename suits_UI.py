import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Set page configuration
st.set_page_config(page_title="Suits for Chicago Housing Laws", page_icon="🏛️", layout="centered")

# Custom CSS for Netflix-style theme
st.markdown(
    """
    <style>
        body {
            background-color: #1A1A1D;
            color: #E0E0E0;
        }
        .stButton>button {
            background-color: #B71C1C;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            font-size: 20px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #E53935;
            color: #000000;
        }
        .stRadio > div {
            display: none;
        }
        .option-button {
            display: inline-block;
            text-align: center;
            margin: 20px;
            padding: 30px;
            border: 2px solid #B71C1C;
            border-radius: 10px;
            background: #1A1A1D;
            color: #E0E0E0;
            font-size: 24px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .option-button:hover {
            background: #E53935;
            color: #000;
            border-color: #E53935;
        }
        hr {
            border: 1px solid #B71C1C;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("🏛️ Suits for Chicago Housing Laws")
st.markdown("<hr>", unsafe_allow_html=True)

# App Description
st.markdown(
    """
    Welcome to **Suits**, your AI-powered legal assistant for Chicago Housing Laws.  
    Choose a tool to get started:
    """,
    unsafe_allow_html=True,
)

# Navigation Buttons
col1, col2, col3, col4 = st.columns(4)
# st.write(st.experimental_query_pages())


with col1:
    if st.button("📄 Document Summarizer"):
        switch_page("Summarizer")

with col2:
    if st.button("🤔 Decision Assistant"):
        switch_page("Streamlit_Suits")

with col3:
    if st.button("🎭 Legal Simulation Studio"):
        switch_page("Legal_Simulation_Studio")
with col4:
    if st.button("🤖 Persona Simulation Studio"):
        switch_page("Persona_Simulator")
# Footer Disclaimer
st.markdown(
    "<hr><small style='color: gray;'>Disclaimer: Suits is for informational purposes only and relies solely on the provided documents. It is not a substitute for professional legal advice.</small>",
    unsafe_allow_html=True,
)
