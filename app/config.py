import sys
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "OnCallAgents"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    database_url: str = "postgresql+psycopg://oncall:oncall@localhost:5432/oncall"

    # RAG settings
    rag_model: str = "claude-sonnet-4-6"
    rag_top_k: int = 3
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # mcp settings
    mcp_monitor_url: str = "http://localhost:8004/mcp"
    mcp_logs_url: str = "http://localhost:8003/mcp"

    @property
    def mcp_servers(self) -> dict:
        return {
            "monitor": {"transport": "streamable_http", "url": self.mcp_monitor_url},
            "logs": {"transport": "streamable_http", "url": self.mcp_logs_url},
        }


settings = Settings()

logger.remove()
logger.add(sys.stderr, level="DEBUG" if settings.debug else "INFO")
