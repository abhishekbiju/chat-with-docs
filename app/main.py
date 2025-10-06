from fastapi import FastAPI
from app.rag_logic import RAGChain, QueryRequest, QueryResponse

app = FastAPI(
    title="Chat with Docs RAG API",
    description="An API for chatting with your documents.",
    version="1.0.0"
)

@app.post("/chat", response_model=QueryResponse)
async def chat_with_docs(request: QueryRequest):
    """
    Endpoint to handle user queries.
    It retrieves context and generates an answer.
    """
    result = await RAGChain.execute(request.query)
    return result

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG System API. Go to /docs for usage."}