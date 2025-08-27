# OCR service module
from .base_ocr import BaseOCRService
from .easyocr_service import EasyOCRService
from .paddleocr_service import PaddleOCRService

__all__ = [
    'BaseOCRService',
    'EasyOCRService',
    'PaddleOCRService'
]
