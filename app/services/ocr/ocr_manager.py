# services/ocr/ocr_manager.py
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
from loguru import logger

from .base_ocr import BaseOCRService, OCRResult
from .easyocr_service import EasyOCRService
from .paddleocr_service import PaddleOCRService


class OCRManager:
    """
    OCR Manager that handles multiple OCR providers with fallback strategies
    """
    
    def __init__(self, **kwargs):
        self.providers: List[BaseOCRService] = []
        self.primary_provider: Optional[BaseOCRService] = None
        self._initialize_providers(**kwargs)
    
    def _initialize_providers(self, **kwargs):
        """Initialize available OCR providers in order of preference"""
        
        # Provider priority order (free local options only)
        provider_classes = [
            EasyOCRService,      # Local, no external dependencies
            PaddleOCRService,    # Local, fast and accurate
        ]
        
        for provider_class in provider_classes:
            try:
                provider = provider_class(**kwargs)
                if provider.is_service_available():
                    self.providers.append(provider)
                    if not self.primary_provider:
                        self.primary_provider = provider
                    logger.info(f"OCR provider initialized: {provider.get_provider_name()}")
                else:
                    logger.warning(f"OCR provider not available: {provider_class.__name__}")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_class.__name__}: {e}")
        
        if not self.providers:
            logger.warning("No OCR providers available")
        else:
            logger.info(f"OCR Manager initialized with {len(self.providers)} providers")
    
    async def extract_text(self, image_path: Path) -> Optional[OCRResult]:
        """Extract text using the best available OCR provider"""
        if not self.providers:
            return None
        
        # Try primary provider first
        if self.primary_provider:
            result = await self.primary_provider.extract_text_with_fallback(image_path)
            if result and result.text.strip():
                return result
        
        # Fallback to other providers
        for provider in self.providers:
            if provider == self.primary_provider:
                continue
            
            result = await provider.extract_text_with_fallback(image_path)
            if result and result.text.strip():
                logger.info(f"Using fallback OCR provider: {provider.get_provider_name()}")
                return result
        
        return None
    
    async def extract_text_from_bytes(self, image_bytes: bytes) -> Optional[OCRResult]:
        """Extract text from bytes using the best available OCR provider"""
        if not self.providers:
            return None
        
        # Try primary provider first
        if self.primary_provider:
            try:
                result = await self.primary_provider.extract_text_from_bytes(image_bytes)
                if result and result.text.strip():
                    return result
            except Exception as e:
                logger.warning(f"Primary OCR provider failed: {e}")
        
        # Fallback to other providers
        for provider in self.providers:
            if provider == self.primary_provider:
                continue
            
            try:
                result = await provider.extract_text_from_bytes(image_bytes)
                if result and result.text.strip():
                    logger.info(f"Using fallback OCR provider: {provider.get_provider_name()}")
                    return result
            except Exception as e:
                logger.warning(f"OCR provider {provider.get_provider_name()} failed: {e}")
        
        return None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available OCR providers"""
        return [provider.get_provider_name() for provider in self.providers]
    
    def is_ocr_available(self) -> bool:
        """Check if any OCR provider is available"""
        return len(self.providers) > 0
    
    async def extract_text_with_retry(self, image_path: Path, max_retries: int = 2) -> Optional[OCRResult]:
        """Extract text with retry logic"""
        for attempt in range(max_retries):
            try:
                result = await self.extract_text(image_path)
                if result and result.text.strip():
                    return result
            except Exception as e:
                logger.warning(f"OCR attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Brief delay before retry
        
        return None
