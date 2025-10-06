import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import httpx
from pydantic import BaseModel

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_API_URL = "http://host.docker.internal:11434/api/chat"
LLM_MODEL_NAME = "mistral" 

# --- Initialize Components ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
client = httpx.AsyncClient(timeout=60.0)

PROMPT_TEMPLATE = """
[INST]
You are a helpful AI assistant. Use the context provided to answer the user's question. If you don't know the answer, just say you don't know. Do not make up an answer.

Context:
{context}

Question:
{question}
[/INST]
"""

class RAGChain:
    @staticmethod
    def retrieve_context(query: str, n_results: int = 3) -> list[str]:
        """Retrieve relevant context from the vector store."""
        results = db.similarity_search(query, k=n_results)
        return [doc.page_content for doc in results]

    @staticmethod
    async def generate_answer(question: str, context: list[str]) -> str:
        """Generate an answer using the Ollama LLM."""
        formatted_context = "\n\n".join(context)
        prompt = PROMPT_TEMPLATE.format(context=formatted_context, question=question)

        payload = {
            "model": LLM_MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        try:
            response = await client.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            
            completion = response.json()['message']['content']
            return completion.strip()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting from Ollama: {e}")
            return "Sorry, I couldn't connect to the language model."

    @classmethod
    async def execute(cls, query: str):
        """Execute the full RAG chain."""
        context = cls.retrieve_context(query)
        answer = await cls.generate_answer(query, context)
        return {"answer": answer, "context": context}

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: list[str]