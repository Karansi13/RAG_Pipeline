# RAG Pipeline - Document Processing and Question Answering System

A comprehensive Retrieval-Augmented Generation (RAG) pipeline that enables intelligent document search and question answering using modern AI technologies.

## 🚀 Overview

This project implements a complete RAG (Retrieval-Augmented Generation) system that can:
- Load and process PDF documents
- Split documents into manageable chunks
- Create vector embeddings for semantic search
- Store embeddings in vector databases
- Answer questions based on document content
- Provide citations and source references
- Stream responses and maintain query history

## 📁 Project Structure

```
RAG_PIPELINE/
├── data/                          # Data storage directory
│   ├── pdf/                       # PDF files for processing
│   │   └── AttentionIsAllYouNeed.pdf
│   ├── text_files/               # Text files
│   │   ├── machine_learning.txt
│   │   └── python_intro.txt
│   └── vector_store/             # Vector database storage
│       ├── chroma.sqlite3
│       └── [vector embeddings]
├── notebook/                     # Jupyter notebooks
│   ├── pdf_loader.ipynb         # PDF processing and RAG implementation
│   └── document.ipynb           # Document structure examples
├── main.py                       # Main application entry point
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
├── .env                         # Environment variables (API keys)
├── .gitignore                   # Git ignore rules
├── .python-version              # Python version specification
└── README.md                    # Project documentation
```

## 🛠️ Technologies Used

### Core Libraries
- **LangChain** - Framework for building LLM applications
- **LangChain Community** - Additional LangChain components
- **LangChain Core** - Core LangChain functionality
- **LangChain Groq** - Groq LLM integration

### Document Processing
- **PyPDF** - PDF text extraction
- **PyMuPDF** - Advanced PDF processing with better text extraction

### Vector Storage & Search
- **ChromaDB** - Vector database for document embeddings
- **FAISS CPU** - Facebook AI Similarity Search for efficient vector operations
- **Sentence Transformers** - Creating high-quality text embeddings

### Environment Management
- **Python-dotenv** - Environment variable management
- **IPyKernel** - Jupyter notebook support

## 🔧 Installation & Setup

### Prerequisites
- Python 3.12 or higher
- Groq API key (for LLM functionality)

### Step 1: Clone and Setup Environment
```bash
# Navigate to project directory
cd RAG_PIPELINE

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# OR install from pyproject.toml
pip install -e .
```

### Step 3: Environment Configuration
Create a `.env` file in the project root and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Quick Start

### 1. Basic Document Processing
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF documents
loader = PyPDFLoader("data/pdf/AttentionIsAllYouNeed.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
```

### 2. Vector Store Creation
```python
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb

# Create embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="data/vector_store")
```

### 3. RAG Query System
```python
# Query the system
question = "What is attention mechanism in transformers?"
results = rag_system.query(question, top_k=3)
print(results['answer'])
```

## 📊 Features

### Document Processing
- **Multi-format Support**: PDF processing with PyPDF and PyMuPDF
- **Intelligent Chunking**: Recursive text splitting with configurable chunk sizes
- **Metadata Preservation**: Maintains source file information, page numbers, and document structure

### Vector Storage
- **ChromaDB Integration**: Persistent vector storage with efficient similarity search
- **FAISS Support**: High-performance similarity search capabilities
- **Embeddings**: Uses Sentence Transformers for semantic text representation

### Advanced RAG Features
- **Streaming Responses**: Real-time answer generation
- **Source Citations**: Automatic citation generation with page references
- **Query History**: Maintains conversation context and previous queries
- **Answer Summarization**: Optional answer summarization for concise responses
- **Configurable Retrieval**: Adjustable similarity thresholds and result counts

### LLM Integration
- **Groq API**: Fast inference with Groq's LLM services
- **Flexible Prompting**: Customizable prompt templates for different use cases
- **Context-Aware**: Maintains conversation context for better responses

## 📖 Usage Examples

### Processing Multiple PDFs
The system can process entire directories of PDF files:
```python
# Process all PDFs in a directory
pdf_dir = "data/pdf/"
all_documents = process_all_pdfs(pdf_dir)
print(f"Processed {len(all_documents)} documents")
```

### Advanced RAG Pipeline
```python
# Initialize advanced RAG with streaming and citations
adv_rag = AdvancedRAGPipeline(retriever, llm)

result = adv_rag.query(
    question="Explain the transformer architecture",
    top_k=5,
    min_score=0.2,
    stream=True,
    summarize=True
)

print("Answer:", result['answer'])
print("Summary:", result['summary'])
print("Sources:", result['sources'])
```

### Document Structure Examples
```python
from langchain_core.documents import Document

# Create custom documents
doc = Document(
    page_content="Your document content here",
    metadata={
        "source": "custom_doc.txt",
        "author": "Your Name",
        "date_created": "2025-01-01"
    }
)
```

## 🔄 Workflow

1. **Document Ingestion**: Load PDFs and text files from the `data/` directory
2. **Text Processing**: Split documents into semantically meaningful chunks
3. **Embedding Generation**: Create vector representations using Sentence Transformers
4. **Vector Storage**: Store embeddings in ChromaDB for fast retrieval
5. **Query Processing**: Accept user questions and retrieve relevant context
6. **Answer Generation**: Use Groq LLM to generate contextual answers
7. **Response Enhancement**: Add citations, summaries, and maintain history

## 🎛️ Configuration

### Chunk Size Settings
```python
# Adjustable text splitting parameters
chunk_size = 1000        # Characters per chunk
chunk_overlap = 200      # Overlap between chunks
```

### Retrieval Parameters
```python
# Search configuration
top_k = 5               # Number of similar chunks to retrieve
min_score = 0.2         # Minimum similarity threshold
```

### LLM Settings
```python
# Model configuration through environment variables
GROQ_API_KEY = "your_api_key"
```

## 📊 Sample Data

The project includes sample documents:
- **AttentionIsAllYouNeed.pdf**: The famous Transformer architecture paper
- **machine_learning.txt**: Machine learning concepts
- **python_intro.txt**: Python programming introduction

## 🔍 Key Components

### PDF Loader (`pdf_loader.ipynb`)
- Processes PDF files with metadata extraction
- Handles multiple documents in batch
- Implements text chunking strategies
- Creates and manages vector stores
- Implements complete RAG pipeline with advanced features

### Document Structure (`document.ipynb`)
- Demonstrates LangChain Document objects
- Shows metadata handling
- Provides examples of document creation

### Main Application (`main.py`)
- Entry point for the application
- Currently contains basic setup (can be extended)

## 🚧 Future Enhancements

- [ ] Web interface for document upload and querying
- [ ] Support for additional document formats (DOCX, TXT, etc.)
- [ ] Advanced filtering and search capabilities
- [ ] Integration with more LLM providers
- [ ] Batch processing capabilities
- [ ] Advanced analytics and query insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your Groq API key is correctly set in the `.env` file
2. **Import Errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
3. **Vector Store Issues**: Check that the `data/vector_store/` directory has proper permissions
4. **PDF Processing Errors**: Ensure PDF files are not corrupted and are readable

### Performance Tips

- Use appropriate chunk sizes for your document types
- Adjust similarity thresholds based on your precision/recall needs
- Consider using FAISS for large-scale vector operations
- Monitor memory usage with large document collections

## 📞 Support

For questions, issues, or contributions, please refer to the project documentation or create an issue in the repository.

---

**Built with ❤️ using LangChain, ChromaDB, and modern AI technologies**