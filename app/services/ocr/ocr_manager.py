# services/ocr/ocr_manager.py
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRManager:
    """
    OCR Manager (Disabled for Memory Optimization)
    OCR functionality has been removed to reduce memory usage.
    Text extraction relies on pre-extracted content from other extractors.
    """
    
    def __init__(self, **kwargs):
        self.providers = []
        self.primary_provider = None
        logger.info("OCR Manager initialized (OCR disabled for memory optimization)")
    
    def _initialize_providers(self, **kwargs):
        """No OCR providers available - OCR disabled for memory optimization"""
        logger.info("No OCR providers available - OCR disabled for memory optimization")
    
    async def extract_text(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """OCR disabled - cannot extract text from images"""
        logger.warning("OCR disabled for memory optimization - cannot extract text from images")
        return None
    
    async def extract_text_from_bytes(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """OCR disabled - cannot extract text from image bytes"""
        logger.warning("OCR disabled for memory optimization - cannot extract text from image bytes")
        return None
    
    def get_available_providers(self) -> List[str]:
        """No OCR providers available"""
        return []
    
    def is_any_provider_available(self) -> bool:
        """No OCR providers available"""
        return False
