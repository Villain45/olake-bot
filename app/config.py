import logging
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


LLM_MODEL_PRIORITY = (
    "google/gemini-2.5-flash-lite",
    "z-ai/glm-4.7-flash",
    "deepseek/deepseek-v4.1-flash",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openrouter_api_key: str = ""
    # Keep the permitted models; try Gemini first to avoid waiting on GLM before it.
    llm_model: str = LLM_MODEL_PRIORITY[0]
    llm_fallback_models: list[str] = list(LLM_MODEL_PRIORITY[1:])
    app_url: str = "http://localhost:8080"
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = Field(default=1200, gt=0)
    llm_timeout_s: float = Field(default=25.0, gt=0)
    llm_json_mode: bool = True
    llm_use_system_instruction: bool = True
    llm_concurrency: int = Field(default=8, gt=0)

    # Qdrant Cloud
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection: str = "olake_docs"

    # CPU models
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = Field(default=384, gt=0)
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    model_cache_dir: str = "./.model_cache"
    inference_threads: int = Field(default=1, gt=0)
    rerank_batch_size: int = Field(default=2, gt=0)

    # Retrieval
    candidates: int = Field(default=30, gt=0)
    top_k: int = Field(default=6, gt=0)
    min_rerank_score: float = Field(default=0.25, ge=0, le=1)
    retrieval_concurrency: int = Field(default=2, gt=0)
    retrieval_recovery_enabled: bool = True
    answer_recovery_enabled: bool = True
    recovery_min_remaining_s: float = Field(default=8.0, gt=0)
    index_refresh_s: float = Field(default=5.0, ge=0)
    index_batch_size: int = Field(default=32, gt=0)
    index_cache_path: str = "./data/embedding_cache.sqlite3"
    index_max_tokens: int = Field(default=384, ge=64, le=384)

    # Web fallback
    web_fallback_enabled: bool = True
    web_timeout_s: float = Field(default=4.0, gt=0)
    web_concurrency: int = Field(default=2, gt=0)
    trusted_domains: list[str] = [
        "olake.io",
        "iceberg.apache.org",
        "postgresql.org",
        "dev.mysql.com",
        "mongodb.com",
        "docs.aws.amazon.com",
        "parquet.apache.org",
        "kafka.apache.org",
    ]

    # API
    allowed_origins: list[str] = ["*"]
    rate_limit: str = "20/minute"
    max_question_chars: int = Field(default=2000, ge=1, le=2000)
    cache_ttl_s: int = Field(default=3600, gt=0)
    refusal_cache_ttl_s: int = Field(default=60, gt=0)
    request_timeout_s: float = Field(default=55.0, gt=0)
    readiness_timeout_s: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def validate_retrieval_limits(self):
        if self.top_k > self.candidates:
            raise ValueError("TOP_K must not exceed CANDIDATES")
        if (self.llm_model, *self.llm_fallback_models) != LLM_MODEL_PRIORITY:
            logging.getLogger("olake.config").warning(
                "Ignoring legacy LLM routing settings; using the three permitted models in priority order"
            )
            self.llm_model = LLM_MODEL_PRIORITY[0]
            self.llm_fallback_models = list(LLM_MODEL_PRIORITY[1:])
        return self

    @property
    def llm_models(self) -> tuple[str, ...]:
        return LLM_MODEL_PRIORITY


@lru_cache
def get_settings() -> Settings:
    return Settings()
