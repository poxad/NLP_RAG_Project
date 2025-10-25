import streamlit as st
from streamlit_option_menu import option_menu

st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        height: 100px;
        font-size: 50px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("### 🚀 NLP RAG Project - Intelligent Document Chatbot")
st.info("Advanced Retrieval-Augmented Generation System for Document Processing and Conversational AI")

# Main NLP RAG Chatbot button - prominently displayed
st.markdown("### 🎯 Main Application")
if st.button("🚀 NLP RAG Chatbot", key="main_rag_button"):
    st.switch_page("pages/🚀_NLP_RAG_Chatbot.py")

st.markdown("---")

# Other chatbot options
st.markdown("### 📚 Other Chatbot Variants")
example_prompts = [
    "📄 PDF Chatbot",
    "🖼️ Image Chatbot",
    "📚 Text Narrative Chatbot"
]

button_cols = st.columns(3)

if button_cols[0].button(example_prompts[0]):
    st.switch_page("pages/📄_PDF_Chatbot.py")
if button_cols[1].button(example_prompts[1]):
    st.switch_page("pages/🖼️_Image_Chatbot.py")
if button_cols[2].button(example_prompts[2]):
    st.switch_page("pages/💬_Narrative_Chatbot.py")


