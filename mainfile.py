from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from dotenv import load_dotenv
import os
import json
import requests
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware


class QueryRequest(BaseModel):
    query_text: str

app = FastAPI()

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

load_dotenv()

# client = chromadb.PersistentClient(path="/chroma/chroma")
# CHROMA_DB_URL = "https://chroma.railway.internal"
CHROMA_DB_URL = "https://chroma-production-1576.up.railway.app"
CHROMA_DB_RAG_URL = "https://chroma-rag-production.up.railway.app/query/"
CHROMA_HEARTBEAT = "https://chroma-production-1576.up.railway.app/api/v1/heartbeat"
CHROMA_AUTH_TOKEN = os.getenv('CHROMA_AUTH_TOKEN')
headers = {"Authorization": f"Bearer {CHROMA_AUTH_TOKEN}"}  

client = chromadb.HttpClient(host=CHROMA_DB_URL, ssl=True, headers=headers)

collection = client.get_collection("new_anisha_db")

@app.post("/query/")
def query_chromadb(query: QueryRequest, n_results: int = 9):
    results = collection.query(query_texts=[query.query_text], n_results=n_results)
    response_list = []
    for i in range(len(results["documents"][0])):
        metadata = results["metadatas"][0][i]  # Get metadata for current item
        
        response_item = {
            "answer": results["documents"][0][i],
            "question": metadata.get("question", ""),
            "links": json.loads(metadata.get("links", "[]")),
            "highlighted_text": json.loads(metadata.get("highlighted_text", "[]"))
        }
        response_list.append(response_item)
    
    return {"results": response_list}


@app.get("/wake", 
    response_model=Dict[str, str],
    summary="Check server status",
    description="Endpoint to check if the Chroma server is awake and responding"
)
async def wake_server():
    """
    Checks if the Chroma DB server is awake and responding to requests.
    
    Returns:
        dict: Server status and additional health metrics
    
    Raises:
        HTTPException: If server is unreachable or returns error
    """
    # try:
    response = requests.get(
        CHROMA_HEARTBEAT, 
        headers=headers,
        timeout=5.0  # 5 second timeout
    )

    # Waking-up RAG server
    response = requests.post(
        CHROMA_DB_RAG_URL, 
        headers=headers,
        json={
            "query_text": "Hello!",
            "n_results": 1
        },
        timeout=5.0  # 5 second timeout
    )
    
    response.raise_for_status()
    
    health_data = response.json()
    chromadb.logger.info(f"Server health check successful: {health_data}")
    
    return {
        "status": "awake",
        "latency_ms": str(response.elapsed.total_seconds() * 1000)
    }
        
    # except requests.Timeout:
    #     # chromadb.logger.error("Server health check timed out")
    #     # raise HTTPException(
    #     #     # status_code=Status.HTTP_504_GATEWAY_TIMEOUT,
    #     #     detail="Server health check timed out"
    #     # )
        
    # except requests.RequestException as e:
    #     chromadb.logger.error(f"Server health check failed: {str(e)}")
    #     raise HTTPException(
    #         # status_code=Status.HTTP_503_SERVICE_UNAVAILABLE,
    #         detail=f"Server is not responding: {str(e)}"
    #     )