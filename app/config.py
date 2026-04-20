import sys
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
                                                                                                
    app_name: str = "OnCallAgents"
    debug: bool = False                                                                           
    host: str = "0.0.0.0"
    port: int = 9900                                                                              

    anthropic_api_key: str                                                      
    voyage_api_key: str
                                                                                                
    milvus_host: str = "localhost"
    milvus_port: int = 19530                                                                      
                
    # RAG settings
    rag_model: str = "claude-sonnet-4-6"
    rag_top_k: int = 3                                                                            
    chunk_max_size: int = 800
    chunk_overlap: int = 100                                                                      
                
                                                                                                
settings = Settings()

logger.remove()
logger.add(sys.stderr, level="DEBUG" if settings.debug else "INFO")