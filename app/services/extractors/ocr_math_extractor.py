# services/extractors/ocr_math_extractor.py
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
# REMOVED: torch and transformers imports for memory optimization
from loguru import logger
import re
import io

from app.models.schemas import Equation, Paragraph, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError


class OCRMathExtractor:
    """
    Lightweight OCR and mathematical content extraction using:
    1. Tesseract OCR (general OCR)
    2. Mathematical formula detection (rule-based)
    
    AI models disabled for memory optimization
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "ocr_math"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check Tesseract availability
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR available")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self.tesseract_available = False
        
        # AI models disabled for memory optimization
        self.nougat_available = False
        logger.info("OCR/Math extractor initialized (AI models disabled for memory optimization)")
    
    async def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text and mathematical content from PDF
        Returns enhanced text and equations
        """
        results = {
            "enhanced_text": [],
            "equations": [],
            "ocr_regions": []
        }
        
        # Check if PDF needs OCR
        needs_ocr = await self._check_needs_ocr(pdf_path)
        
        if needs_ocr:
            logger.info("PDF appears to be scanned or contains non-selectable text")
            
            # Method 1: Tesseract OCR (lightweight)
            if self.tesseract_available:
                try:
                    ocr_results = await self._extract_with_tesseract(pdf_path)
                    results["ocr_regions"].extend(ocr_results)
                    logger.info(f"Tesseract processed {len(ocr_results)} regions")
                except Exception as e:
                    logger.error(f"Tesseract extraction failed: {e}")
        
        # Always extract mathematical formulas (even from text PDFs)
        try:
            math_equations = await self._extract_math_formulas(pdf_path)
            results["equations"].extend(math_equations)
            logger.info(f"Extracted {len(math_equations)} additional equations")
        except Exception as e:
            logger.error(f"Math extraction failed: {e}")
        
        # Deduplicate equations
        results["equations"] = self._deduplicate_equations(results["equations"])
        
        return results
    
    async def _check_needs_ocr(self, pdf_path: Path) -> bool:
        """Check if PDF needs OCR processing"""
        doc = fitz.open(str(pdf_path))
        needs_ocr = False
        
        # Sample first few pages
        sample_pages = min(3, len(doc))
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            
            # Check if page has selectable text
            text = page.get_text().strip()
            
            # Get page as image to check content
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            
            # Check if page has visual content but no/little text
            if self._has_visual_content(img_data) and len(text) < 100:
                needs_ocr = True
                break
        
        doc.close()
        return needs_ocr
    
    def _has_visual_content(self, img_data: bytes) -> bool:
        """Check if image has significant visual content"""
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Check if image is not mostly white
        mean_val = np.mean(img)
        return mean_val < 250  # Not a blank page
    
    async def _extract_with_tesseract(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract content using Tesseract OCR"""
        results = []
        
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Convert page to image (lower resolution for memory optimization)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # Reduced from 2x to 1.5x
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Process with Tesseract
            try:
                # Extract text with Tesseract
                text = pytesseract.image_to_string(pil_image, lang='eng')
                
                if text.strip():
                    results.append({
                    "page": page_num,
                        "text": text.strip(),
                        "type": "tesseract_ocr",
                        "confidence": 0.7  # Default confidence for Tesseract
                    })
                    
                    # Look for mathematical content in the extracted text
                    math_content = self._extract_math_from_text(text)
                    if math_content:
                        results.append({
                            "page": page_num,
                            "text": math_content,
                            "type": "math_ocr",
                            "confidence": 0.6
                        })
                
            except Exception as e:
                logger.warning(f"Tesseract failed for page {page_num}: {e}")
        
        doc.close()
        return results
    
    def _extract_math_from_text(self, text: str) -> str:
        """Extract mathematical content from text using pattern matching"""
        math_patterns = [
            r'\$[^$]+\$',  # LaTeX inline math
            r'\\\([^)]+\\\)',  # LaTeX display math
            r'\\\[[^\]]+\\\]',  # LaTeX display math
            r'[a-zA-Z]\s*=\s*[0-9+\-*/()]+',  # Simple equations
            r'[0-9]+\s*[+\-*/]\s*[0-9]+',  # Basic arithmetic
            r'[a-zA-Z]+\s*\([^)]+\)',  # Function calls
        ]
        
        math_content = []
        for pattern in math_patterns:
            matches = re.findall(pattern, text)
            math_content.extend(matches)
        
        return '\n'.join(math_content) if math_content else ""
    
    async def _extract_math_formulas(self, pdf_path: Path) -> List[Equation]:
        """Extract mathematical formulas from PDF"""
        equations = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text from page
                text = page.get_text()
                
                # Look for mathematical patterns
                math_patterns = [
                    r'\$[^$]+\$',  # LaTeX inline math
                    r'\\\([^)]+\\\)',  # LaTeX display math
                    r'\\\[[^\]]+\\\]',  # LaTeX display math
                    r'[a-zA-Z]\s*=\s*[0-9+\-*/()]+',  # Simple equations
                    r'[0-9]+\s*[+\-*/]\s*[0-9]+',  # Basic arithmetic
                ]
                
                for pattern in math_patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        equation_text = match.group(0)
                        
                        # Create equation object
                        equation = Equation(
                            id=f"eq_page{page_num + 1}_{len(equations)}",
                            page=page_num + 1,
                            text=equation_text,
                            type="latex" if "$" in equation_text or "\\" in equation_text else "text",
                            confidence=0.8,
                            bounding_box=BoundingBox(
                                x=0, y=0, width=100, height=50  # Default values
                            )
                        )
                        equations.append(equation)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Math formula extraction failed: {e}")
        
        return equations
    
    def _deduplicate_equations(self, equations: List[Equation]) -> List[Equation]:
        """Remove duplicate equations"""
        unique_equations = []
        seen_text = set()
        
        for equation in equations:
            if equation.text not in seen_text:
                seen_text.add(equation.text)
                unique_equations.append(equation)
        
        return unique_equations