# services/extractors/ocr_math_extractor.py
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel
from loguru import logger
import re
import io

from app.models.schemas import Equation, Paragraph, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError


class OCRMathExtractor:
    """
    OCR and mathematical content extraction using:
    1. Nougat (for academic PDFs with math)
    2. Tesseract (general OCR)
    3. Mathematical formula detection
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "ocr_math"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        self._init_models()
    
    def _init_models(self):
        """Initialize Nougat and other models"""
        try:
            # Initialize Nougat
            self.nougat_processor = NougatProcessor.from_pretrained(
                settings.nougat_model_path
            )
            self.nougat_model = VisionEncoderDecoderModel.from_pretrained(
                settings.nougat_model_path
            ).to(self.device)
            self.nougat_model.eval()
            self.nougat_available = True
            logger.info("Nougat model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Nougat model: {e}")
            self.nougat_available = False
        
        # Check Tesseract availability
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR available")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self.tesseract_available = False
    
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
            
            # Method 1: Nougat (best for academic papers with math)
            if self.nougat_available:
                try:
                    nougat_results = await self._extract_with_nougat(pdf_path)
                    results["enhanced_text"].extend(nougat_results["text"])
                    results["equations"].extend(nougat_results["equations"])
                    logger.info(f"Nougat extracted {len(nougat_results['equations'])} equations")
                except Exception as e:
                    logger.error(f"Nougat extraction failed: {e}")
            
            # Method 2: Tesseract OCR (fallback for general text)
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
    
    async def _extract_with_nougat(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract content using Nougat model"""
        results = {
            "text": [],
            "equations": []
        }
        
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Process with Nougat
            try:
                # Prepare image
                pixel_values = self.nougat_processor(
                    images=pil_image,
                    return_tensors="pt"
                ).pixel_values.to(self.device)
                
                # Generate output
                with torch.no_grad():
                    outputs = self.nougat_model.generate(
                        pixel_values,
                        min_length=1,
                        max_new_tokens=3584,
                        bad_words_ids=[[self.nougat_processor.tokenizer.unk_token_id]]
                    )
                
                # Decode output
                sequence = self.nougat_processor.batch_decode(
                    outputs, skip_special_tokens=True
                )[0]
                
                # Parse markdown/LaTeX content
                parsed = self._parse_nougat_output(sequence, page_num)
                
                results["text"].append({
                    "page": page_num,
                    "text": parsed["text"],
                    "type": "nougat_ocr"
                })
                
                results["equations"].extend(parsed["equations"])
                
            except Exception as e:
                logger.warning(f"Nougat failed for page {page_num}: {e}")
        
        doc.close()
        return results
    
    def _parse_nougat_output(self, markdown_text: str, page_num: int) -> Dict[str, Any]:
        """Parse Nougat markdown output to extract text and equations"""
        result = {
            "text": "",
            "equations": []
        }
        
        # Extract display equations ($$...$$)
        display_pattern = r'\$\$(.*?)\$\$'
        display_matches = re.finditer(display_pattern, markdown_text, re.DOTALL)
        
        for match in display_matches:
            latex = match.group(1).strip()
            if latex:
                equation = Equation(
                    latex=latex,
                    page=page_num,
                    inline=False
                )
                result["equations"].append(equation)
        
        # Extract inline equations ($...$)
        inline_pattern = r'\$([^\$]+?)\$'
        inline_matches = re.finditer(inline_pattern, markdown_text)
        
        for match in inline_matches:
            latex = match.group(1).strip()
            if latex and len(latex) > 1:  # Skip single characters
                equation = Equation(
                    latex=latex,
                    page=page_num,
                    inline=True
                )
                result["equations"].append(equation)
        
        # Clean text (remove math markup)
        clean_text = re.sub(r'\$\$.*?\$\$', '[EQUATION]', markdown_text, flags=re.DOTALL)
        clean_text = re.sub(r'\$[^\$]+?\$', '[FORMULA]', clean_text)
        
        # Remove markdown formatting
        clean_text = re.sub(r'#+\s*', '', clean_text)  # Headers
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)  # Bold
        clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)  # Italic
        
        result["text"] = clean_text.strip()
        
        return result
    
    async def _extract_with_tesseract(self, pdf_path: Path) -> List[Dict]:
        """Extract text using Tesseract OCR"""
        results = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Preprocess image for better OCR
            processed_img = self._preprocess_for_ocr(img)
            
            # Run OCR with detailed output
            try:
                # Get detailed OCR data
                ocr_data = pytesseract.image_to_data(
                    processed_img,
                    lang=settings.ocr_language,
                    output_type=pytesseract.Output.DICT
                )
                
                # Group text by blocks
                blocks = self._group_ocr_blocks(ocr_data, page_num, page.rect)
                results.extend(blocks)
                
                # Also run specialized math OCR if available
                math_regions = self._detect_math_regions(processed_img)
                for region in math_regions:
                    math_text = self._ocr_math_region(processed_img, region, page_num)
                    if math_text:
                        results.append(math_text)
                
            except Exception as e:
                logger.warning(f"Tesseract failed for page {page_num}: {e}")
        
        doc.close()
        return results
    
    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(
            denoised, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Deskew if needed
        angle = self._get_skew_angle(binary)
        if abs(angle) > 0.5:
            binary = self._rotate_image(binary, angle)
        
        return binary
    
    def _get_skew_angle(self, img: np.ndarray) -> float:
        """Detect skew angle of text"""
        # Find all white pixels
        coords = np.column_stack(np.where(img > 0))
        
        if len(coords) < 100:
            return 0.0
        
        # Fit a line using PCA
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        return angle
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image to correct skew"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _group_ocr_blocks(self, ocr_data: Dict, page_num: int, page_rect) -> List[Dict]:
        """Group OCR results into logical blocks"""
        blocks = []
        current_block = None
        
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) < 30:  # Low confidence
                continue
            
            text = ocr_data['text'][i].strip()
            if not text:
                continue
            
            # Get position
            x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                         ocr_data['width'][i], ocr_data['height'][i])
            
            # Check if new block
            block_num = ocr_data['block_num'][i]
            
            if current_block is None or current_block['block_num'] != block_num:
                if current_block:
                    blocks.append(current_block)
                
                current_block = {
                    'block_num': block_num,
                    'page': page_num,
                    'text': text,
                    'bbox': BoundingBox(
                        x1=x, y1=y, x2=x+w, y2=y+h,
                        page=page_num,
                        confidence=ocr_data['conf'][i] / 100.0
                    ),
                    'type': 'tesseract_ocr'
                }
            else:
                # Append to current block
                current_block['text'] += ' ' + text
                # Expand bbox
                current_block['bbox'].x2 = max(current_block['bbox'].x2, x + w)
                current_block['bbox'].y2 = max(current_block['bbox'].y2, y + h)
        
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _detect_math_regions(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions likely containing mathematical formulas"""
        regions = []
        
        # Use edge detection to find formula regions
        edges = cv2.Canny(img, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        height, width = img.shape[:2]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Heuristics for math regions
            aspect_ratio = w / h if h > 0 else 0
            area_ratio = (w * h) / (width * height)
            
            # Math formulas often have specific characteristics
            if (0.1 < aspect_ratio < 10 and  # Not too tall or wide
                0.001 < area_ratio < 0.1 and  # Reasonable size
                w > 30 and h > 10):  # Minimum size
                
                # Check for math-like symbols
                roi = img[y:y+h, x:x+w]
                if self._contains_math_symbols(roi):
                    regions.append((x, y, w, h))
        
        return regions
    
    def _contains_math_symbols(self, img: np.ndarray) -> bool:
        """Check if image region likely contains mathematical symbols"""
        # Simple heuristic: check for specific patterns
        # This could be enhanced with a trained classifier
        
        # Run OCR to check for math-like characters
        try:
            text = pytesseract.image_to_string(img, config='--psm 8')
            math_chars = set('∫∑∏√±×÷≈≠≤≥∞αβγδεζηθικλμνξπρστυφχψω')
            return any(c in math_chars for c in text)
        except:
            return False
    
    def _ocr_math_region(self, img: np.ndarray, region: Tuple, page_num: int) -> Optional[Dict]:
        """OCR a specific region containing math"""
        x, y, w, h = region
        roi = img[y:y+h, x:x+w]
        
        try:
            # Use different OCR settings for math
            text = pytesseract.image_to_string(
                roi,
                config='--psm 8 -c tessedit_char_whitelist=0123456789+-=*/^()[]{}ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            )
            
            if text.strip():
                return {
                    'text': text.strip(),
                    'page': page_num,
                    'bbox': BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h, page=page_num),
                    'type': 'math_ocr'
                }
        except:
            pass
        
        return None
    
    async def _extract_math_formulas(self, pdf_path: Path) -> List[Equation]:
        """Extract mathematical formulas from PDF text"""
        equations = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            
            # Look for LaTeX-style math
            # Display equations
            display_matches = re.finditer(r'\\begin{equation}(.*?)\\end{equation}', text, re.DOTALL)
            for match in display_matches:
                equations.append(Equation(
                    latex=match.group(1).strip(),
                    page=page_num,
                    inline=False
                ))
            
            # Inline math with \( \)
            inline_matches = re.finditer(r'\\\((.*?)\\\)', text)
            for match in inline_matches:
                equations.append(Equation(
                    latex=match.group(1).strip(),
                    page=page_num,
                    inline=True
                ))
            
            # Alternative patterns
            # Numbered equations
            numbered_matches = re.finditer(r'(?:^|\n)\s*([A-Za-z0-9\s\+\-\*\/\=\(\)]+)\s*\(\d+\)\s*(?:\n|$)', text)
            for match in numbered_matches:
                formula = match.group(1).strip()
                if self._is_likely_equation(formula):
                    equations.append(Equation(
                        latex=formula,
                        page=page_num,
                        inline=False
                    ))
        
        doc.close()
        return equations
    
    def _is_likely_equation(self, text: str) -> bool:
        """Check if text is likely a mathematical equation"""
        # Simple heuristics
        math_operators = ['=', '+', '-', '*', '/', '^', '∑', '∫', '∏', '√']
        return any(op in text for op in math_operators) and len(text) > 3
    
    def _deduplicate_equations(self, equations: List[Equation]) -> List[Equation]:
        """Remove duplicate equations"""
        seen = set()
        unique = []
        
        for eq in equations:
            # Use normalized latex as key
            key = self._normalize_latex(eq.latex)
            if key not in seen:
                seen.add(key)
                unique.append(eq)
        
        return unique
    
    def _normalize_latex(self, latex: str) -> str:
        """Normalize LaTeX for comparison"""
        # Remove whitespace
        normalized = re.sub(r'\s+', '', latex)
        # Remove optional braces
        normalized = re.sub(r'[{}]', '', normalized)
        return normalized.lower()