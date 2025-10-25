import time
import os
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
from PIL import Image
import easyocr
import numpy as np
import pdf2image
from io import BytesIO
import base64

load_dotenv()

# Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
api_key = OPENAI_API_KEY
new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '🚀'

# Create directories
os.makedirs('data/', exist_ok=True)
os.makedirs('faiss_index/', exist_ok=True)

# Load past chats
try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}


def get_file_text(files):
    """Enhanced text extraction from multiple file types with file tracking"""
    text = ""
    reader = easyocr.Reader(['en'])
    
    # Store file information for later use
    st.session_state.uploaded_files = []
    
    for file in files:
        file_type = file.name.split('.')[-1].lower()
        file_info = {
            'name': file.name,
            'type': file_type,
            'size': file.size
        }
        st.session_state.uploaded_files.append(file_info)
        
        text += f"\n--- Content from {file.name} (File Type: {file_type.upper()}, Size: {file.size} bytes) ---\n"
        
        if file_type == 'pdf':
            # Handle PDF files
            try:
                pdf_reader = PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += f"[Page {page_num} of {file.name}]\n{extracted_text}\n"
            except:
                # Fallback to OCR for scanned PDFs
                file.seek(0)
                images = pdf2image.convert_from_bytes(file.read())
                for page_num, page in enumerate(images, 1):
                    text += f"[Page {page_num} of {file.name} - OCR Text]\n"
                    results = reader.readtext(np.array(page))
                    for result in results:
                        text += result[1] + " "
                    text += "\n"
        
        elif file_type in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
            # Handle image files
            try:
                file.seek(0)
                image = Image.open(file)
                text += f"[Image: {file.name} - OCR Text]\n"
                results = reader.readtext(np.array(image))
                for result in results:
                    text += result[1] + " "
                text += "\n"
            except Exception as e:
                text += f"[Error processing image {file.name}: {str(e)}]\n"
    
    return text

def get_uploaded_files_info():
    """Get information about uploaded files"""
    try:
        # First try to get from session state
        if hasattr(st.session_state, 'uploaded_files') and st.session_state.uploaded_files:
            files_info = []
            for file_info in st.session_state.uploaded_files:
                files_info.append(f"{file_info['name']} ({file_info['type'].upper()}, {file_info['size']} bytes)")
            return files_info
        
        # Fallback to FAISS index
        if os.path.exists("faiss_index/index.pkl"):
            import pickle
            with open("faiss_index/index.pkl", "rb") as f:
                index_data = pickle.load(f)
                if hasattr(index_data, 'docstore') and hasattr(index_data.docstore, '_dict'):
                    files_info = []
                    file_names = set()
                    for doc_id, doc in index_data.docstore._dict.items():
                        if hasattr(doc, 'metadata') and 'file_name' in doc.metadata:
                            file_names.add(doc.metadata['file_name'])
                    for file_name in file_names:
                        files_info.append(file_name)
                    return files_info if files_info else ["Files processed but names not available"]
    except Exception as e:
        pass
    return ["No file information available"]

def get_text_chunks(text):
    """Enhanced text chunking with better parameters"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Smaller chunks for better precision
        chunk_overlap=200,  # Better overlap for context
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Enhanced vector store with OpenAI embeddings and file metadata"""
    # Use OpenAI embeddings - better quality and generous free tier
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-small"  # Latest and most efficient OpenAI embedding model
    )
    
    # Create metadata with file information
    metadatas = []
    for i, chunk in enumerate(text_chunks):
        # Extract file name from chunk if it contains file information
        file_name = "Unknown"
        if "--- Content from" in chunk:
            try:
                file_name = chunk.split("--- Content from ")[1].split(" (")[0]
            except:
                pass
        
        metadatas.append({
            "source": f"chunk_{i}",
            "file_name": file_name,
            "chunk_index": i
        })
    
    vector_store = FAISS.from_texts(
        text_chunks, 
        embedding=embeddings,
        metadatas=metadatas
    )
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """Enhanced prompt template for better RAG performance with file awareness and chat history"""
    prompt_template = """
    You are an intelligent assistant with access to a comprehensive knowledge base from uploaded documents and previous conversation history. Your role is to provide accurate, detailed, and contextually relevant answers based on the provided context.

    IMPORTANT: 
    - The context contains information from specific uploaded files. Each piece of content is clearly marked with the source file name and page number (for PDFs) or image name (for images).
    - You have access to the previous conversation history to understand the context of ongoing discussions.

    INSTRUCTIONS:
    1. Analyze the provided context carefully, paying attention to file sources
    2. Consider the previous conversation history to understand the context of the current question
    3. Answer the question using information from the context and previous conversation
    4. When referencing information, mention which file it came from (e.g., "According to [filename.pdf]" or "From the image [filename.jpg]")
    5. If the context doesn't contain enough information, clearly state this
    6. Provide specific examples or quotes from the context when relevant
    7. Structure your response clearly with proper formatting
    8. If multiple sources are available, synthesize information from all relevant sources
    9. Always acknowledge the source files when providing information
    10. Reference previous parts of the conversation when relevant

    CONTEXT (from uploaded files):
    {context}

    QUESTION:
    {question}

    RESPONSE:
    """
    
    model = ChatOpenAI(
        model="gpt-4o-mini",  # Latest and most efficient OpenAI model
        temperature=0.1,  # Lower temperature for more consistent responses
        openai_api_key=api_key,
        max_tokens=2048
    )
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    """Enhanced user input processing with OpenAI embeddings and streaming"""
    try:
        # Handle special queries about uploaded files
        if any(phrase in user_question.lower() for phrase in ["uploaded files", "files uploaded", "what files", "which files", "file names"]):
            files_info = get_uploaded_files_info()
            if files_info and files_info[0] != "No file information available":
                return f"Here are the uploaded files: {', '.join(files_info)}"
            else:
                return "No files have been uploaded and processed yet. Please upload files using the sidebar and click 'Submit & Process'."
        
        # Check if FAISS index exists
        if not os.path.exists("faiss_index/index.faiss"):
            # If no documents, provide a general response
            return f"I'm doing well, thank you for asking! I'm here to help you with any questions you might have. If you'd like me to answer questions about specific documents, please upload them using the sidebar and click 'Submit & Process'. Otherwise, feel free to ask me anything!"
        
        # Use OpenAI embeddings for better quality and consistency
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small"
        )
        
        # Load FAISS index with error handling
        try:
            new_db = FAISS.load_local("faiss_index", embeddings)
        except Exception as faiss_error:
            return f"Error loading document index: {str(faiss_error)}. Please re-upload and process your documents."
        
        # Enhanced similarity search with more results
        docs = new_db.similarity_search(
            user_question, 
            k=5,  # Get more relevant documents
            fetch_k=20  # Consider more candidates
        )
        
        if not docs:
            return "No relevant information found in the uploaded documents. Please try a different question or upload more documents."
        
        # Additional metadata-based filtering
        docs_with_scores = new_db.similarity_search_with_score(user_question, k=5)
        
        # Prepare context with chat history
        context_with_history = ""
        
        # Add recent chat history (last 5 messages) to context
        if hasattr(st.session_state, 'messages') and len(st.session_state.messages) > 1:
            context_with_history += "\n--- PREVIOUS CONVERSATION HISTORY ---\n"
            recent_messages = st.session_state.messages[-6:-1]  # Last 5 messages (excluding current)
            for msg in recent_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    context_with_history += f"User: {content}\n"
                else:
                    context_with_history += f"Assistant: {content}\n"
            context_with_history += "\n--- END CONVERSATION HISTORY ---\n\n"
        
        # Add document context
        context_with_history += "--- DOCUMENT CONTEXT ---\n"
        for doc in docs:
            context_with_history += doc.page_content + "\n\n"
        
        # Create a proper document object for the chain
        from langchain.schema import Document
        context_doc = Document(page_content=context_with_history, metadata={})
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": [context_doc], "question": user_question}, 
            return_only_outputs=True
        )
        return response["output_text"]
    
    except Exception as e:
        return f"I apologize, but I encountered an error processing your request: {str(e)}. Please try again or re-upload your documents."

@st.dialog("Clear chat history?")
def modal():
    button_cols = st.columns([1, 1])  # Equal column width for compact fit
    
    # Add custom CSS for button styling
    st.markdown(
        """
        <style>
        .stButton button {
            width: 100%;
            padding: 10px;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if button_cols[0].button("Yes"):
        clear_chat_history()
        st.rerun()
    elif button_cols[1].button("No"):
        st.rerun()

def clear_chat_history():
    """Clear all chat history and data"""
    st.session_state.pop('chat_id', None)
    st.session_state.pop('messages', None)
    st.session_state.pop('gemini_history', None)
    
    # Clear data files
    for file in Path('data/').glob('*'):
        file.unlink()
    
    # Clear FAISS index
    for file in Path('faiss_index/').glob('*'):
        file.unlink()

# st.write('# Enhanced Multimodal RAG Chatbot')

# Sidebar
with st.sidebar:
    st.write('# Sidebar Menu')

    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    if st.button("Clear Chat History", key="clear_chat_button"):
        modal()

    uploaded_files = st.file_uploader("Upload your PDF and Image Files (Auto-processes on upload)", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp'])
    
    # Auto-process files immediately upon upload
    if uploaded_files:
        with st.spinner("🔄 Processing files automatically..."):
            try:
                raw_text = get_file_text(uploaded_files)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)!")
                    st.balloons()
                else:
                    st.error("❌ No text could be extracted from the uploaded files.")
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
 
    # Save new chats after a message has been sent to AI
    # Set chat title with current date and time
    st.session_state.chat_title = f'Enhanced-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'

st.write('# NLP RAG Chatbot')

# Load chat history
try:
    st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
    st.session_state.gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []

# Initialize OpenAI model
st.session_state.model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=api_key,
    temperature=0.1
)
st.session_state.chat = st.session_state.model

# Check if 'messages' is not in st.session_state and initialize with a default message
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "✨",  # or any valid emoji
        "content": "Hey there, I'm your Enhanced Multimodal chatbot. Please upload the necessary files in the sidebar to add more context to this conversation."
    })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message.get('role', 'user'),
        avatar=message.get('avatar', None),
    ):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input('Your message here...'):
    # Display user message in chat message container
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )

    with st.spinner("Waiting for AI response..."):
        response = user_input(prompt, api_key)
        
    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=response,
            avatar=AI_AVATAR_ICON,
        )
    )
    # OpenAI doesn't maintain history in the same way, so we keep our own
    # st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')

# print(st.session_state)
