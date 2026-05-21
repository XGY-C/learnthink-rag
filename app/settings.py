from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Knowledge base
    kb_base_dir: Path = Field(default=Path("../kb"), alias="KB_BASE_DIR")
    excerpt_max_chars: int = Field(default=320, alias="EXCERPT_MAX_CHARS")

    # Milvus — 两种连接方式（二选一）
    # 方式 A: Milvus Lite 嵌入式（开发首选，零外部依赖）
    milvus_uri: str | None = Field(default=None, alias="MILVUS_URI")
    # 方式 B: Milvus Standalone（Docker / 远程服务器）
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_db: str = Field(default="learnthink", alias="MILVUS_DB")

    # Embedding
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_cache_dir: str = Field(default="./models", alias="EMBEDDING_CACHE_DIR")

    # Hybrid search
    search_mode: str = Field(default="hybrid", alias="SEARCH_MODE")  # dense | sparse | hybrid
    sparse_weight: float = Field(default=0.3, alias="SPARSE_WEIGHT")  # 稀疏权重，仅 hybrid 模式生效

    # Alibaba Cloud OSS
    oss_access_key: str = Field(default="", alias="OSS_ACCESS_KEY")
    oss_secret_key: str = Field(default="", alias="OSS_SECRET_KEY")
    oss_endpoint: str = Field(default="oss-cn-beijing.aliyuncs.com", alias="OSS_ENDPOINT")
    oss_bucket_name: str = Field(default="learn-think", alias="OSS_BUCKET_NAME")


settings = Settings()
