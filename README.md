# Chat with Your Docs

This project is a locally-hosted application that allows you to chat with your own PDF documents. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers based on the content you provide. The tech stack includes FastAPI, Ollama, LangChain, and ChromaDB.

---

## Features

* **Local Data Ingestion:** Process and use your own PDF files as a knowledge base.
* **Private & Local LLM:** Runs entirely on your machine using Ollama, so your data stays private.
* **Simple REST API:** A straightforward `/chat` endpoint for easy integration and testing.
* **Dockerized:** The application is containerized for a quick and reliable setup.

---

## How to Use (Local Setup)

Follow these steps to get the project running on your local machine.

1.  **Clone the Repository**
    Open your terminal and clone the project files.
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Add Your Documents**
    Place your PDF files inside the `/documents` directory.

3.  **Install Dependencies**
    Install the necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Ollama**
    In a new terminal, pull the model and start the Ollama server. Leave this terminal running in the background.
    ```bash
    ollama pull mistral
    ollama serve
    ```

5.  **Ingest Data**
    Run the ingestion script to process your PDFs into the vector database.
    ```bash
    python app/ingest.py
    ```

6.  **Start the Application**
    Use Docker Compose to build and run the API server.
    ```bash
    docker-compose up --build
    ```
    The API will be available at `http://localhost:8080`. You can test it easily by visiting the interactive docs at `http://localhost:8080/docs`.

---

## Future Improvements

Here are a few ideas for extending the project:

* **Web Interface:** Build a simple front-end using a framework like Streamlit or Gradio to create a user-friendly chat interface.
* **Streaming Responses:** Modify the API to stream the LLM's response token-by-token for a more interactive experience.
* **Support More File Types:** Add support for ingesting other document formats like `.docx`, `.txt`, and `.md`.
* **Chat History:** Implement conversation memory so the model can remember previous parts of your conversation.
* **Advanced RAG:** Explore more advanced RAG techniques like query transformations or result re-ranking to improve the quality of the answers.