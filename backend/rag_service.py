import requests
import json
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any
import os
import time
from models import DocumentChunk


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.max_retries = 3
        self.retry_delay = 2

    def _check_ollama_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _check_model_availability(self, model_name: str) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model_name in model.get("name", "") for model in models)
            return False
        except:
            return False

    def generate_embeddings(self, text: str) -> List[float]:
        if not self._check_ollama_connection():
            raise Exception("Ollama service is not running. Please start Ollama with 'ollama serve'")

        if not self._check_model_availability("nomic-embed-text"):
            raise Exception("Model 'nomic-embed-text' not found. Please run 'ollama pull nomic-embed-text'")

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": "nomic-embed-text",
                        "prompt": text
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["embedding"]
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Error generating embeddings after {self.max_retries} attempts: {str(e)}")
                time.sleep(self.retry_delay)

        raise Exception("Failed to generate embeddings")

    def chat_completion(self, messages: List[Dict], context: str = "") -> str:
        if not self._check_ollama_connection():
            raise Exception("Ollama service is not running. Please start Ollama with 'ollama serve'")

        if not self._check_model_availability("tinyllama"):
            raise Exception("Model 'tinyllama:1.1b' not found. Please run 'ollama pull tinyllama:1.1b'")

        try:
            system_prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
            Use the context to answer the user's question accurately. If the answer cannot be found in the context, say so.

            Context:
            {context}
            """

            user_message = messages[-1]["content"] if messages else ""

            full_prompt = f"{system_prompt}\n\nUser Question: {user_message}"

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "tinyllama:1.1b",
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise Exception(f"Error in chat completion: {str(e)}")


class RAGService:
    def __init__(self, persist_directory: str = "./vector_db"):
        self.ollama_client = OllamaClient()
        self.persist_directory = persist_directory

        os.makedirs(persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        try:
            self.collection = self.chroma_client.get_collection(name="pdf_documents")
        except:
            self.collection = self.chroma_client.create_collection(name="pdf_documents")

        self.chat_sessions = {}

    def health_check(self) -> Dict[str, str]:
        status = {
            "ollama_connection": "disconnected",
            "nomic_embed_model": "not_available",
            "tinyllama_model": "not_available",
            "vector_db": "connected"
        }

        try:
            if self.ollama_client._check_ollama_connection():
                status["ollama_connection"] = "connected"

                if self.ollama_client._check_model_availability("nomic-embed-text"):
                    status["nomic_embed_model"] = "available"

                if self.ollama_client._check_model_availability("tinyllama"):
                    status["tinyllama_model"] = "available"
        except:
            pass

        return status

    def add_documents(self, chunks: List[dict]) -> int:
        try:
            documents = []
            metadatas = []
            ids = []
            embeddings = []

            for i, chunk in enumerate(chunks):
                # Generate embedding for each chunk
                embedding = self.ollama_client.generate_embeddings(chunk['content'])

                documents.append(chunk['content'])
                metadatas.append(chunk['metadata'])
                ids.append(f"{chunk['metadata']['filename']}_{chunk['metadata']['chunk_id']}_{i}")
                embeddings.append(embedding)

            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            return len(chunks)
        except Exception as e:
            raise Exception(f"Error adding documents to vector DB: {str(e)}")

    def similarity_search(self, query: str, k: int = 5) -> List[DocumentChunk]:
        try:

            query_embedding = self.ollama_client.generate_embeddings(query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            chunks = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    chunk = DocumentChunk(
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        similarity_score=1 - results['distances'][0][i] if results['distances'] else None
                    )
                    chunks.append(chunk)

            return chunks
        except Exception as e:
            raise Exception(f"Error in similarity search: {str(e)}")

    def get_chat_history(self, session_id: str) -> List[Dict]:
        return self.chat_sessions.get(session_id, [])

    def add_to_chat_history(self, session_id: str, role: str, content: str):
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = []

        self.chat_sessions[session_id].append({
            "role": role,
            "content": content
        })

        if len(self.chat_sessions[session_id]) > 10:
            self.chat_sessions[session_id] = self.chat_sessions[session_id][-10:]

    def chat_with_rag(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        try:
            relevant_chunks = self.similarity_search(message, k=3)

            context = "\n\n".join([chunk.content for chunk in relevant_chunks])

            chat_history = self.get_chat_history(session_id)

            self.add_to_chat_history(session_id, "user", message)

            response = self.ollama_client.chat_completion(
                messages=self.get_chat_history(session_id),
                context=context
            )

            self.add_to_chat_history(session_id, "assistant", response)

            sources = []
            for chunk in relevant_chunks:
                source_info = f"File: {chunk.metadata.get('filename', 'Unknown')}"
                if 'page' in chunk.metadata:
                    source_info += f", Page: {chunk.metadata['page']}"
                sources.append(source_info)

            return {
                "response": response,
                "sources": sources,
                "session_id": session_id
            }
        except Exception as e:
            raise Exception(f"Error in RAG chat: {str(e)}")