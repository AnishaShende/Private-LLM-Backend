from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, BaseSettings, Field
import chromadb
from dotenv import load_dotenv
import os
import json
import httpx
import logging
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup proper logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration management
class Settings(BaseSettings):
    chroma_db_url: str = "https://chroma-production-1576.up.railway.app"
    chroma_db_rag_url: str = "https://chroma-rag-production.up.railway.app/query/"
    chroma_heartbeat: str = "https://chroma-production-1576.up.railway.app/api/v1/heartbeat"
    chroma_auth_token: str
    collection_name: str = "new_anisha_db"
    environment: str = "development"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

# Pydantic models
class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1, max_length=1000, description="The query text to search for")
    n_results: Optional[int] = Field(default=9, ge=1, le=50, description="Number of results to return")

class QueryResponse(BaseModel):
    answer: str
    question: str
    links: List[str]
    highlighted_text: List[str]

class QueryResult(BaseModel):
    results: List[QueryResponse]
    total_results: int
    query_time_ms: float

class HealthResponse(BaseModel):
    status: str
    latency_ms: float
    timestamp: str
    chroma_status: str = "unknown"

# ChromaDB Service
class ChromaDBService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.headers = {"Authorization": f"Bearer {settings.chroma_auth_token}"}
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self._client = None
        self._collection = None
        
    def get_client(self):
        if self._client is None:
            try:
                self._client = chromadb.HttpClient(
                    host=self.settings.chroma_db_url,
                    ssl=True,
                    headers=self.headers
                )
                logger.info("ChromaDB client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to connect to ChromaDB"
                )
        return self._client
    
    def get_collection(self):
        if self._collection is None:
            try:
                client = self.get_client()
                self._collection = client.get_collection(self.settings.collection_name)
                logger.info(f"Collection '{self.settings.collection_name}' loaded successfully")
            except Exception as e:
                logger.error(f"Failed to get collection: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to access ChromaDB collection"
                )
        return self._collection
    
    async def query_collection(self, query_text: str, n_results: int) -> QueryResult:
        start_time = time.time()
        
        try:
            collection = self.get_collection()
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            response_list = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    metadata = results["metadatas"][0][i]
                    
                    # Safely parse JSON fields
                    try:
                        links = json.loads(metadata.get("links", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        links = []
                    
                    try:
                        highlighted_text = json.loads(metadata.get("highlighted_text", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        highlighted_text = []
                    
                    response_item = QueryResponse(
                        answer=results["documents"][0][i],
                        question=metadata.get("question", ""),
                        links=links,
                        highlighted_text=highlighted_text
                    )
                    response_list.append(response_item)
            
            query_time = (time.time() - start_time) * 1000
            
            return QueryResult(
                results=response_list,
                total_results=len(response_list),
                query_time_ms=round(query_time, 2)
            )
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Query processing failed"
            )
    
    async def health_check(self) -> HealthResponse:
        start_time = time.time()
        chroma_status = "unknown"
        
        try:
            # Check ChromaDB heartbeat
            response = await self.http_client.get(
                self.settings.chroma_heartbeat,
                headers=self.headers,
                timeout=5.0
            )
            response.raise_for_status()
            chroma_status = "healthy"
            
            # Optionally wake up RAG server (non-blocking)
            try:
                await self.http_client.post(
                    self.settings.chroma_db_rag_url,
                    headers=self.headers,
                    json={"query_text": "Hello!", "n_results": 1},
                    timeout=5.0
                )
            except Exception as e:
                logger.warning(f"RAG server wake-up failed: {str(e)}")
            
            latency = (time.time() - start_time) * 1000
            
            return HealthResponse(
                status="healthy",
                latency_ms=round(latency, 2),
                timestamp=datetime.utcnow().isoformat(),
                chroma_status=chroma_status
            )
            
        except httpx.TimeoutException:
            logger.error("Health check timed out")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Health check timed out"
            )
        except httpx.RequestError as e:
            logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service unavailable: {str(e)}"
            )
    
    async def close(self):
        if self.http_client:
            await self.http_client.aclose()

@lru_cache()
def get_chroma_service(settings: Settings = Depends(get_settings)):
    return ChromaDBService(settings)

# FastAPI app initialization
app = FastAPI(
    title="ChromaDB Query API",
    description="API for querying ChromaDB collections",
    version="1.0.0"
)

# CORS configuration
settings = get_settings()
cors_origins = settings.cors_origins
if settings.environment == "production":
    # In production, specify exact origins
    cors_origins = [origin for origin in cors_origins if not origin.startswith("http://localhost")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up ChromaDB Query API...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"CORS origins: {cors_origins}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ChromaDB Query API...")

# API endpoints
@app.post("/query/", response_model=QueryResult)
async def query_chromadb(
    query: QueryRequest,
    chroma_service: ChromaDBService = Depends(get_chroma_service)
):
    """
    Query the ChromaDB collection with the provided text.
    
    - **query_text**: The text to search for in the collection
    - **n_results**: Number of results to return (1-50, default: 9)
    """
    logger.info(f"Processing query: '{query.query_text[:50]}...' with n_results={query.n_results}")
    
    return await chroma_service.query_collection(
        query_text=query.query_text,
        n_results=query.n_results
    )

@app.get("/wake", response_model=HealthResponse)
async def wake_server(
    chroma_service: ChromaDBService = Depends(get_chroma_service)
):
    """
    Check if the Chroma server is awake and responding to requests.
    
    Returns server status and health metrics.
    """
    logger.info("Performing health check...")
    return await chroma_service.health_check()

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Simple health check endpoint for the API itself.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)