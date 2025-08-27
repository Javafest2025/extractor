# services/ocr/paddleocr_service.py
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import io
from loguru import logger

from .base_ocr import BaseOCRService, OCRResult


class PaddleOCRService(BaseOCRService):
    """
    PaddleOCR service - fast and accurate OCR from Baidu
    No external dependencies like Tesseract required
    """
    
    def __init__(self, **kwargs):
        self.ocr = None
        self.languages = kwargs.get('languages', ['en'])
        super().__init__(**kwargs)
    
    def _initialize_service(self, **kwargs) -> bool:
        """Initialize PaddleOCR service"""
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=kwargs.get('use_gpu', False)
            )
            self.is_available = True
            logger.info("PaddleOCR initialized successfully")
            return True
        except ImportError:
            logger.warning("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            return False
    
    async def extract_text(self, image_path: Path) -> Optional[OCRResult]:
        """Extract text from image file using PaddleOCR"""
        try:
            # Run OCR
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.ocr.ocr, str(image_path), cls=True
            )
            
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return None
    
    async def extract_text_from_bytes(self, image_bytes: bytes) -> Optional[OCRResult]:
        """Extract text from image bytes using PaddleOCR"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Run OCR
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.ocr.ocr, image, cls=True
            )
            
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction from bytes failed: {e}")
            return None
    
    def _process_results(self, results: List[List]) -> OCRResult:
        """Process PaddleOCR results into OCRResult format"""
        if not results or not results[0]:
            return OCRResult(text="", confidence=0.0, provider="PaddleOCR")
        
        # Extract text and confidence scores
        texts = []
        confidences = []
        words = []
        
        for line in results[0]:
            if len(line) >= 2:
                bbox = line[0]  # Bounding box coordinates
                text_info = line[1]
                
                if len(text_info) >= 2:
                    text = text_info[0]
                    confidence = text_info[1]
                    
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
            return OCRResult(text="", confidence=0.0, provider="PaddleOCR")
        
        # Combine text and calculate average confidence
        combined_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences)
        
        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            language='en',
            words=words,
            provider="PaddleOCR"
        )
