# 🚀 Enhanced Multimodal RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that supports multiple file types, advanced AI models, and intelligent document processing.

## ✨ Key Features

### 🔄 **Multi-Modal File Support**

- **PDF Documents**: Direct text extraction with OCR fallback for scanned PDFs
- **Image Files**: OCR text extraction from PNG, JPG, JPEG, GIF, BMP
- **Batch Processing**: Upload multiple files simultaneously
- **Smart Processing**: Automatic file type detection and appropriate handling

### 🤖 **Advanced AI Integration**

- **OpenAI GPT-4o-mini**: Latest and most efficient model for text generation
- **OpenAI Embeddings**: High-quality `text-embedding-3-small` for vector search
- **No Quota Limits**: Unlike Google's embedding API, OpenAI provides generous limits
- **Cost-Effective**: Optimized for both performance and cost

### 🧠 **Intelligent RAG System**

- **Enhanced Similarity Search**: Advanced parameters (k=5, fetch_k=20) for better relevance
- **File Source Attribution**: AI references specific files when providing information
- **Context-Aware Responses**: Combines document content with conversation history
- **Smart Chunking**: Optimized text splitting (1500 chars, 200 overlap) for precision

### 💬 **Conversational Memory**

- **Chat History Integration**: AI remembers previous conversations
- **Context Continuity**: Follows the flow of ongoing discussions
- **Session Management**: Multiple chat sessions with easy switching
- **Persistent Storage**: Chat history saved across app restarts

### 🎯 **User Experience**

- **Streamlit Interface**: Clean, modern UI with sidebar controls
- **File Management**: Easy upload, process, and track uploaded documents
- **Error Handling**: Graceful error messages and helpful guidance
- **Real-time Processing**: Live feedback during document processing

## 🛠️ Technical Architecture

### **RAG Pipeline**

```
Document Upload → Text Extraction → Chunking → Embeddings → Vector Store → Similarity Search → LLM Generation
```

### **File Processing Flow**

1. **Upload**: Multi-file support with type validation
2. **Extract**: Text extraction with OCR fallback
3. **Chunk**: Smart text splitting for optimal retrieval
4. **Embed**: OpenAI embeddings for vector representation
5. **Index**: FAISS vector store for fast similarity search
6. **Query**: Enhanced search with metadata filtering

### **AI Integration**

- **LLM**: OpenAI GPT-4o-mini for response generation
- **Embeddings**: OpenAI text-embedding-3-small for vector search
- **Vector Store**: FAISS for efficient similarity search
- **Chain**: LangChain for orchestrated RAG workflow

## 📋 Installation

### **Prerequisites**

- Python 3.8+
- OpenAI API Key

### **Setup**

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd chatbot-rag-streamlit
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment setup**

   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run pages/🚀_Enhanced_Multimodal_Chatbot.py
   ```

## 🚀 Usage

### **Basic Workflow**

1. **Upload Files**: Drag and drop PDFs or images in the sidebar
2. **Process Documents**: Click "Submit & Process" to build the knowledge base
3. **Start Chatting**: Ask questions about your documents
4. **Switch Sessions**: Use the sidebar to manage different conversations

### **Advanced Features**

- **File Queries**: Ask "What files did I upload?" to see processed documents
- **Source Attribution**: AI will mention which file contains the information
- **Context Awareness**: AI remembers previous questions and conversations
- **Multi-Session**: Switch between different chat sessions

## 📁 Project Structure

```
chatbot-rag-streamlit/
├── pages/
│   ├── 🚀_Enhanced_Multimodal_Chatbot.py  # Main enhanced chatbot
│   ├── 📄_PDF_Chatbot.py                  # Original PDF chatbot
│   ├── 🖼️_Image_Chatbot.py                # Image chatbot
│   └── 💬_Narrative_Chatbot.py            # Narrative chatbot
├── data/                                  # Chat history storage
├── faiss_index/                          # Vector store files
├── requirements.txt                       # Dependencies
└── README.md                             # This file
```

## 🔧 Configuration

### **Environment Variables**

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### **Model Settings**

- **LLM Model**: `gpt-4o-mini` (latest, most efficient)
- **Embedding Model**: `text-embedding-3-small`
- **Temperature**: 0.1 (for consistent responses)
- **Max Tokens**: 2048

### **RAG Parameters**

- **Chunk Size**: 1500 characters
- **Chunk Overlap**: 200 characters
- **Similarity Search**: k=5, fetch_k=20
- **Context History**: Last 5 messages

## 🎯 Key Improvements Over Original

| Feature                | Original              | Enhanced                  |
| ---------------------- | --------------------- | ------------------------- |
| **File Types**         | PDF only              | PDF + Images              |
| **AI Model**           | Gemini Pro            | GPT-4o-mini               |
| **Embeddings**         | Google (quota issues) | OpenAI (no limits)        |
| **Memory**             | Basic                 | Full conversation history |
| **Source Attribution** | None                  | File-specific references  |
| **Error Handling**     | Limited               | Comprehensive             |
| **User Experience**    | Basic                 | Advanced with streaming   |

## 🚀 Advanced Features

### **Intelligent Document Processing**

- **OCR Fallback**: Automatic OCR for scanned PDFs
- **Multi-page Support**: Handles complex PDF documents
- **Image Text Extraction**: OCR from various image formats
- **Metadata Tracking**: File names, types, and sizes preserved

### **Enhanced RAG Capabilities**

- **Context-Aware Search**: Combines document content with chat history
- **Source Attribution**: AI references specific files and pages
- **Smart Retrieval**: Advanced similarity search with metadata filtering
- **Conversational Memory**: Maintains context across interactions

### **User Interface**

- **Modern Design**: Clean Streamlit interface
- **Session Management**: Multiple chat sessions
- **File Tracking**: Detailed information about uploaded files
- **Error Recovery**: Graceful handling of processing errors

## 🔍 Troubleshooting

### **Common Issues**

1. **No API Key**: Ensure `OPENAI_API_KEY` is set in `.env`
2. **Import Errors**: Run `pip install -r requirements.txt`
3. **File Processing**: Check file formats are supported
4. **Memory Issues**: Clear chat history if needed

### **Performance Optimization**

- **Chunk Size**: Adjust for your document types
- **Search Parameters**: Tune k and fetch_k values
- **Model Selection**: Choose appropriate OpenAI models
- **Caching**: FAISS index persists between sessions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI**: For GPT-4o-mini and embedding models
- **LangChain**: For RAG framework and tools
- **Streamlit**: For the web interface
- **FAISS**: For vector similarity search
- **EasyOCR**: For image text extraction

---

**Built with ❤️ for intelligent document processing and conversational AI**
