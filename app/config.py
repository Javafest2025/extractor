# app/config.py
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Application
    app_name: str = "PDFExtractor"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8002
    api_prefix: str = "/api/v1"
    
    # External Services
    grobid_url: str = "http://localhost:8070"
    pdffigures2_path: str = ""  # Disabled - using fallback methods (PyMuPDF + CV)
    nougat_model_path: str = "facebook/nougat-base"
    
    # Enrichment APIs
    crossref_api_url: str = "https://api.crossref.org"
    crossref_email: Optional[str] = None
    openalex_api_url: str = "https://api.openalex.org"
    unpaywall_api_url: str = "https://api.unpaywall.org/v2"
    unpaywall_email: Optional[str] = None
    
    # Storage
    paper_folder: Path = Path("./paper")
    output_format: str = "json"
    keep_intermediate_files: bool = True
    
    # Processing
    max_workers: int = 4
    extraction_timeout: int = 300
    ocr_language: str = "eng"
    use_gpu: bool = True
    
    # Cache
    redis_url: Optional[str] = None
    cache_ttl: int = 3600
    
    # RabbitMQ
    rabbitmq_user: Optional[str] = None
    rabbitmq_password: Optional[str] = None
    
    # PDF Storage (B2)
    b2_key_id: Optional[str] = None
    b2_application_key: Optional[str] = None
    b2_bucket_name: Optional[str] = None
    b2_bucket_id: Optional[str] = None
    
    # Gemini API
    gemini_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create paper folder if it doesn't exist
        self.paper_folder.mkdir(exist_ok=True)


# Create global settings instance
settings = Settings()