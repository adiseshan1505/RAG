from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from typing import List
import shutil

from models import ChatMessage, ChatResponse, UploadResponse
from rag_service import RAGService
from pdf_processor import PDFProcessor

# Initialize FastAPI app
app = FastAPI(title="RAG PDF System", description="RAG system for PDF documents using Ollama")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = RAGService()
pdf_processor = PDFProcessor()

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "RAG PDF System API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "disconnected"
    except:
        ollama_status = "disconnected"

    return {
        "status": "healthy",
        "ollama": ollama_status,
        "vector_db": "connected"
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read and process PDF
        with open(file_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()

        # Extract text from PDF
        text = pdf_processor.extract_text_from_pdf(pdf_content)

        # Create chunks
        chunks = pdf_processor.chunk_text(text, file.filename)

        # Add to vector database
        chunks_created = rag_service.add_documents(chunks)

        return UploadResponse(
            message="PDF uploaded and processed successfully",
            filename=file.filename,
            chunks_created=chunks_created
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    """Chat with the RAG system"""
    try:
        result = rag_service.chat_with_rag(
            message=chat_message.message,
            session_id=chat_message.session_id
        )

        return ChatResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")


@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    history = rag_service.get_chat_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.delete("/sessions/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    if session_id in rag_service.chat_sessions:
        del rag_service.chat_sessions[session_id]
    return {"message": f"Chat history cleared for session: {session_id}"}


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        # Get all documents from ChromaDB
        results = rag_service.collection.get()

        # Extract unique filenames
        filenames = set()
        if results['metadatas']:
            for metadata in results['metadatas']:
                if 'filename' in metadata:
                    filenames.add(metadata['filename'])

        return {"documents": list(filenames)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)