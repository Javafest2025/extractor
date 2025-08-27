# services/ocr/base_ocr.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
from dataclasses import dataclass


@dataclass
class OCRResult:
    """Result from OCR processing"""
    text: str
    confidence: float
    language: Optional[str] = None
    words: Optional[List[Dict[str, Any]]] = None
    provider: str = "unknown"


class BaseOCRService(ABC):
    """
    Base class for OCR services
    All OCR providers should implement this interface
    """
    
    def __init__(self, **kwargs):
        self.provider_name = self.__class__.__name__
        self.is_available = False
        self._initialize_service(**kwargs)
    
    @abstractmethod
    def _initialize_service(self, **kwargs) -> bool:
        """Initialize the OCR service"""
        pass
    
    @abstractmethod
    async def extract_text(self, image_path: Path) -> Optional[OCRResult]:
        """Extract text from image file"""
        pass
    
    @abstractmethod
    async def extract_text_from_bytes(self, image_bytes: bytes) -> Optional[OCRResult]:
        """Extract text from image bytes"""
        pass
    
    def is_service_available(self) -> bool:
        """Check if the OCR service is available"""
        return self.is_available
    
    def get_provider_name(self) -> str:
        """Get the OCR provider name"""
        return self.provider_name
    
    async def extract_text_with_fallback(self, image_path: Path) -> Optional[OCRResult]:
        """Extract text with error handling and fallback"""
        try:
            if not self.is_available:
                return None
            
            result = await self.extract_text(image_path)
            return result
            
        except Exception as e:
            # Log error but don't fail the entire extraction
            from loguru import logger
            logger.warning(f"OCR extraction failed with {self.provider_name}: {e}")
            return None
