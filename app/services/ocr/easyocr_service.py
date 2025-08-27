# services/ocr/easyocr_service.py
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import io
from loguru import logger

from .base_ocr import BaseOCRService, OCRResult


class EasyOCRService(BaseOCRService):
    """
    EasyOCR service - lightweight OCR that doesn't require Tesseract
    Uses deep learning models for text recognition
    """
    
    def __init__(self, **kwargs):
        self.reader = None
        self.languages = kwargs.get('languages', ['en'])
        super().__init__(**kwargs)
    
    def _initialize_service(self, **kwargs) -> bool:
        """Initialize EasyOCR service"""
        try:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=kwargs.get('use_gpu', False))
            self.is_available = True
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
            return True
        except ImportError:
            logger.warning("EasyOCR not available. Install with: pip install easyocr")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            return False
    
    async def extract_text(self, image_path: Path) -> Optional[OCRResult]:
        """Extract text from image file using EasyOCR"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Run OCR
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.reader.readtext, image
            )
            
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return None
    
    async def extract_text_from_bytes(self, image_bytes: bytes) -> Optional[OCRResult]:
        """Extract text from image bytes using EasyOCR"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Run OCR
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.reader.readtext, image
            )
            
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"EasyOCR extraction from bytes failed: {e}")
            return None
    
    def _process_results(self, results: List[tuple]) -> OCRResult:
        """Process EasyOCR results into OCRResult format"""
        if not results:
            return OCRResult(text="", confidence=0.0, provider="EasyOCR")
        
        # Extract text and confidence scores
        texts = []
        confidences = []
        words = []
        
        for (bbox, text, confidence) in results:
            if text.strip() and confidence > 0.1:  # Filter low confidence results
                texts.append(text.strip())
                confidences.append(confidence)
                
                # Store word-level information
                words.append({
                    'text': text.strip(),
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        if not texts:
            return OCRResult(text="", confidence=0.0, provider="EasyOCR")
        
        # Combine text and calculate average confidence
        combined_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences)
        
        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            language='en',
            words=words,
            provider="EasyOCR"
        )
