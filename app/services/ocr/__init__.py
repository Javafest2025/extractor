# OCR service module
from .base_ocr import BaseOCRService
from .easyocr_service import EasyOCRService

__all__ = [
    'BaseOCRService',
    'EasyOCRService'
]
