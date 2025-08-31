# services/extractors/figure_extractor.py
import subprocess
import json
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from loguru import logger
import httpx
from dataclasses import dataclass
import io

from app.models.schemas import Figure, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError
from app.services.ocr.ocr_manager import OCRManager
from app.services.cloudinary_service import cloudinary_service


@dataclass
class FigureCandidate:
    """Intermediate representation for figure candidates"""
    bbox: BoundingBox
    confidence: float
    method: str
    image_path: Optional[str] = None
    caption: Optional[str] = None
    label: Optional[str] = None
    page: int = 1
    ocr_text: Optional[str] = None  # OCR extracted text from the image
    ocr_confidence: Optional[float] = None  # OCR confidence score


class FigureExtractor:
    """
    Enhanced multi-method figure extraction using PDFFigures2 as primary
    and computer vision techniques as fallback with advanced validation
    and OCR text extraction for LLM processing
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "figures"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.pdffigures2_available = self._check_pdffigures2()
        
        # Initialize OCR manager
        self.ocr_manager = OCRManager(use_gpu=settings.use_gpu)
        
        # Caption detection patterns
        self.CAPTION_PATTERNS = [
            r'(?:Figure|Fig\.?)\s*(\d+)[:.]?\s*([^.]*(?:\.[^.]*){0,2})',  # Figure 1: Caption
            r'(?:Figure|Fig\.?)\s*(\d+)\s*[-–—]\s*([^.]*(?:\.[^.]*){0,2})',  # Figure 1 - Caption
            r'(\d+)\.\s*([^.]*(?:\.[^.]*){0,2})',  # 1. Caption (when near figure)
        ]
        
        # Figure validation thresholds
        self.MIN_FIGURE_AREA = 5000  # Minimum area in pixels²
        self.MAX_FIGURE_AREA = 500000  # Maximum area in pixels²
        self.MIN_ASPECT_RATIO = 0.2
        self.MAX_ASPECT_RATIO = 5.0
    
    def _check_pdffigures2(self) -> bool:
        """Check if PDFFigures2 is available"""
        # If path is empty, PDFFigures2 is disabled
        if not settings.pdffigures2_path:
            logger.info("PDFFigures2 disabled, using fallback methods")
            return False
            
        try:
            # Check if it's a Docker command
            if settings.pdffigures2_path.startswith("docker"):
                result = subprocess.run(
                    settings.pdffigures2_path.split() + ["-h"],
                    capture_output=True,
                    timeout=10
                )
            else:
                result = subprocess.run(
                    [settings.pdffigures2_path, "-h"],
                    capture_output=True,
                    timeout=5
                )
            return result.returncode == 0
        except:
            logger.warning("PDFFigures2 not available, using fallback methods")
            return False
    
    def _is_ocr_available(self):
        """Check if OCR is available"""
        return self.ocr_manager.is_ocr_available()
    
    async def extract(self, pdf_path: Path) -> List[Figure]:
        """
        Enhanced figure extraction with validation and caption detection
        """
        candidates = []
        
        # Phase 1: Extract candidates from multiple methods
        extraction_methods = [
            ('pdffigures2', self._extract_with_pdffigures2),
            ('cv_detection', self._extract_with_cv_enhanced)
        ]
        
        for method_name, method_func in extraction_methods:
            if method_name == 'pdffigures2' and not self.pdffigures2_available:
                continue
                
            try:
                method_candidates = await method_func(pdf_path)
                candidates.extend(method_candidates)
                logger.info(f"{method_name} extracted {len(method_candidates)} figure candidates")
            except Exception as e:
                logger.error(f"{method_name} extraction failed: {e}")
        
        # Phase 2: Enhanced caption detection
        candidates_with_captions = await self._detect_captions(candidates, pdf_path)
        
        # Phase 3: Figure validation and filtering
        validated_candidates = await self._validate_figure_candidates(candidates_with_captions, pdf_path)
        
        # Phase 4: OCR text extraction for LLM processing
        ocr_enhanced_candidates = []
        for candidate in validated_candidates:
            # Try to extract OCR from saved image first
            if candidate.image_path:
                candidate = await self._extract_ocr_from_figure(candidate, pdf_path)
            else:
                # Fallback to extracting from PDF region
                candidate = await self._extract_ocr_from_pdf_region(candidate, pdf_path)
            ocr_enhanced_candidates.append(candidate)
        
        # Phase 5: Reference matching with text
        enhanced_candidates = await self._match_text_references(ocr_enhanced_candidates, pdf_path)
        
        # Phase 6: Deduplication and ranking
        final_figures = self._deduplicate_and_rank_figures(enhanced_candidates)
        
        logger.info(f"Final result: {len(final_figures)} validated figures from {len(candidates)} candidates")
        
        return final_figures
    
    async def _extract_with_pdffigures2(self, pdf_path: Path) -> List[FigureCandidate]:
        """Enhanced PDFFigures2 extraction with better error handling"""
        output_prefix = self.output_dir / pdf_path.stem
        
        # Run PDFFigures2
        if settings.pdffigures2_path.startswith("docker"):
            cmd = settings.pdffigures2_path.split() + [
                str(pdf_path),
                "-m", str(output_prefix),
                "-d", str(self.output_dir),
                "-j", str(output_prefix) + ".json"
            ]
        else:
            cmd = [
                settings.pdffigures2_path,
                str(pdf_path),
                "-m", str(output_prefix),
                "-d", str(self.output_dir),
                "-j", str(output_prefix) + ".json"
            ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise ExtractionError(f"PDFFigures2 failed: {stderr.decode()}")
        
        # Parse JSON output
        json_path = Path(str(output_prefix) + ".json")
        if not json_path.exists():
            return []
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        candidates = []
        for fig_data in data:
            # Enhanced data extraction
            region_boundary = fig_data.get('regionBoundary', {})
            
            candidate = FigureCandidate(
                bbox=BoundingBox(
                    x1=region_boundary.get('x1', 0),
                    y1=region_boundary.get('y1', 0),
                    x2=region_boundary.get('x2', 0),
                    y2=region_boundary.get('y2', 0),
                    page=fig_data.get('page', 1),
                    confidence=0.9  # High confidence for PDFFigures2
                ),
                confidence=0.9,
                method='pdffigures2',
                image_path=fig_data.get('renderURL', ''),
                caption=fig_data.get('caption', ''),
                label=fig_data.get('name', ''),
                page=fig_data.get('page', 1)
            )
            candidates.append(candidate)
        
        return candidates
    

    
    async def _extract_with_cv(self, pdf_path: Path) -> List[FigureCandidate]:
        """
        Legacy CV extraction method - now calls enhanced version
        """
        return await self._extract_with_cv_enhanced(pdf_path)
    
    def _is_likely_figure(self, img: np.ndarray) -> bool:
        """
        Legacy method - now uses enhanced validation
        """
        return self._is_likely_figure_content(img)
    
    def _deduplicate_figures(self, 
                           existing: List[Figure], 
                           new: List[Figure],
                           iou_threshold: float = 0.5) -> List[Figure]:
        """
        Legacy deduplication method - now uses enhanced version
        """
        # Convert to candidates for enhanced deduplication
        existing_candidates = []
        new_candidates = []
        
        # Convert existing figures to candidates
        for fig in existing:
            candidate = FigureCandidate(
                bbox=fig.bbox,
                confidence=fig.confidence or 0.5,
                method=fig.extraction_method or 'unknown',
                image_path=fig.image_path,
                caption=fig.caption,
                label=fig.label,
                page=fig.page
            )
            existing_candidates.append(candidate)
        
        # Convert new figures to candidates
        for fig in new:
            candidate = FigureCandidate(
                bbox=fig.bbox,
                confidence=fig.confidence or 0.5,
                method=fig.extraction_method or 'unknown',
                image_path=fig.image_path,
                caption=fig.caption,
                label=fig.label,
                page=fig.page
            )
            new_candidates.append(candidate)
        
        # Use enhanced deduplication
        unique_candidates = self._remove_duplicate_candidates(new_candidates, iou_threshold)
        
        # Convert back to figures
        unique_figures = []
        for candidate in unique_candidates:
            figure = Figure(
                label=candidate.label,
                caption=candidate.caption,
                page=candidate.page,
                bbox=candidate.bbox,
                image_path=candidate.image_path,
                type=self._determine_figure_type(candidate),
                extraction_method=candidate.method,
                confidence=candidate.confidence,
                ocr_text=candidate.ocr_text,
                ocr_confidence=candidate.ocr_confidence
            )
            unique_figures.append(figure)
        
        return unique_figures
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        # Calculate intersection
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


    
    def _is_likely_figure_content(self, img: np.ndarray) -> bool:
        """Enhanced heuristic to determine if image contains figure content"""
        if img is None or img.size == 0:
            return False
        
        height, width = img.shape[:2]
        
        # Size validation
        area = width * height
        if area < self.MIN_FIGURE_AREA or area > self.MAX_FIGURE_AREA:
            return False
        
        # Aspect ratio validation
        aspect_ratio = width / height
        if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
            return False
        
        # Convert to grayscale for analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Content analysis
        score = 0
        
        # 1. Edge density (figures typically have structured content)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        if 0.02 <= edge_ratio <= 0.3:  # Reasonable edge density
            score += 1
        
        # 2. Variance (figures have variation, not uniform backgrounds)
        variance = np.var(gray)
        if variance > 200:  # Sufficient variation
            score += 1
        
        # 3. Color diversity (figures often have multiple colors)
        if len(img.shape) == 3:
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            if unique_colors > 50:  # Diverse color palette
                score += 1
        
        # 4. Text content analysis (figures shouldn't be mostly text)
        text_likelihood = self._estimate_text_content(gray)
        if text_likelihood < 0.7:  # Not predominantly text
            score += 1
        
        # 5. Geometric shapes detection
        if self._detect_geometric_patterns(gray):
            score += 1
        
        # Require at least 3 positive indicators
        return score >= 3
    
    def _estimate_text_content(self, gray_img: np.ndarray) -> float:
        """Estimate likelihood that image contains primarily text"""
        # Look for text-like patterns
        # Horizontal lines (text baselines)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Vertical spacing patterns typical of text
        vertical_projection = np.sum(horizontal_lines, axis=1)
        
        # Count peaks (potential text lines)
        peaks = []
        for i in range(1, len(vertical_projection) - 1):
            if (vertical_projection[i] > vertical_projection[i-1] and 
                vertical_projection[i] > vertical_projection[i+1] and
                vertical_projection[i] > np.mean(vertical_projection)):
                peaks.append(i)
        
        # Regular spacing suggests text
        if len(peaks) >= 3:
            spacings = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
            spacing_variance = np.var(spacings) if spacings else float('inf')
            
            # Low variance in line spacing suggests text
            if spacing_variance < 50:  # Relatively consistent line spacing
                return 0.8
        
        return 0.3  # Low text likelihood
    
    def _detect_geometric_patterns(self, gray_img: np.ndarray) -> bool:
        """Detect geometric patterns common in figures"""
        # Detect lines
        edges = cv2.Canny(gray_img, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None and len(lines) > 5:
            return True
        
        # Detect circles
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None and len(circles[0]) > 0:
            return True
        
        # Detect rectangles/contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = 0
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Count rectangular shapes
            if len(approx) == 4:
                rectangles += 1
        
        return rectangles >= 3
    
    async def _extract_with_cv_enhanced(self, pdf_path: Path) -> List[FigureCandidate]:
        """Enhanced computer vision detection with better filtering"""
        candidates = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Render page to high-resolution image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale figure detection
            candidates_page = await self._multi_scale_figure_detection(
                img, gray, page_num, page
            )
            candidates.extend(candidates_page)
        
        doc.close()
        return candidates
    
    async def _multi_scale_figure_detection(self, img: np.ndarray, gray: np.ndarray, 
                                          page_num: int, page) -> List[FigureCandidate]:
        """Multi-scale approach for figure detection"""
        candidates = []
        height, width = img.shape[:2]
        
        # Method 1: Contour-based detection
        contour_candidates = await self._detect_figures_by_contours(img, gray, page_num, page)
        candidates.extend(contour_candidates)
        
        # Method 2: Template matching for common figure patterns
        template_candidates = await self._detect_figures_by_templates(img, gray, page_num, page)
        candidates.extend(template_candidates)
        
        # Method 3: Connected component analysis
        component_candidates = await self._detect_figures_by_components(img, gray, page_num, page)
        candidates.extend(component_candidates)
        
        return candidates
    
    async def _detect_figures_by_contours(self, img: np.ndarray, gray: np.ndarray, 
                                  page_num: int, page) -> List[FigureCandidate]:
        """Detect figures using contour analysis"""
        candidates = []
        
        # Enhanced edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Morphological operations to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = img.shape[:2]
        page_area = width * height
        
        for idx, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size and proportion
            if (self.MIN_FIGURE_AREA < area < min(self.MAX_FIGURE_AREA, page_area * 0.4) and
                self.MIN_ASPECT_RATIO < w/h < self.MAX_ASPECT_RATIO):
                
                # Expand boundary to capture full figure including captions and labels
                expanded_bbox = self._expand_figure_boundary(img, gray, x, y, w, h, page_area)
                if expanded_bbox:
                    x, y, w, h = expanded_bbox
                
                # Extract region for validation
                roi = img[y:y+h, x:x+w]
                
                if self._validate_figure_region(roi):
                    # CV contour images are good quality - upload them
                    image_path = None
                    if settings.store_locally:
                        # Store locally in organized folder structure
                        cv_contour_dir = self.output_dir / "figures" / "cv_contour"
                        cv_contour_dir.mkdir(parents=True, exist_ok=True)
                        img_path = cv_contour_dir / f"page{page_num}_fig{idx}.png"
                        cv2.imwrite(str(img_path), roi)
                        image_path = str(img_path)
                        logger.debug(f"Stored CV contour image locally: {image_path}")
                    else:
                        # Upload to Cloudinary
                        filename = f"cv_contour_page{page_num}_fig{idx}"
                        image_path = await cloudinary_service.upload_cv_image(roi, filename, "scholarai/figures")
                        logger.debug(f"Uploaded CV contour image to Cloudinary: {image_path}")
                    
                    # Convert coordinates to PDF space
                    scale = page.rect.width / width
                    
                    candidate = FigureCandidate(
                        bbox=BoundingBox(
                            x1=x * scale,
                            y1=y * scale,
                            x2=(x + w) * scale,
                            y2=(y + h) * scale,
                            page=page_num,
                            confidence=0.7
                        ),
                        confidence=0.7,
                        method='cv_contour',
                        image_path=image_path,
                        label=f"Figure {page_num}.{idx}",
                        page=page_num
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _expand_figure_boundary(self, img: np.ndarray, gray: np.ndarray, 
                               x: int, y: int, w: int, h: int, page_area: int) -> Optional[tuple]:
        """Intelligently expand figure boundary to capture captions, labels, and sub-components"""
        height, width = img.shape[:2]
        
        # Initial expansion factors
        expand_x = int(w * 0.2)  # 20% horizontal expansion
        expand_y = int(h * 0.3)  # 30% vertical expansion (more for captions)
        
        # Calculate new boundaries
        new_x = max(0, x - expand_x)
        new_y = max(0, y - expand_y)
        new_w = min(width - new_x, w + 2 * expand_x)
        new_h = min(height - new_y, h + 2 * expand_y)
        
        # Look for text content around the figure
        text_regions = self._detect_text_regions_around_figure(gray, x, y, w, h)
        
        # Expand to include detected text regions
        for tx, ty, tw, th in text_regions:
                            # Check if text region is close to the figure
            if self._is_text_region_near_figure(tx, ty, tw, th, x, y, w, h):
                new_x = min(new_x, tx)
                new_y = min(new_y, ty)
                new_w = max(new_w, tx + tw - new_x)
                new_h = max(new_h, ty + th - new_y)
        
        # Look for connected components that might be part of the figure
        connected_regions = self._find_connected_figure_components(gray, x, y, w, h)
        
        # Expand to include connected components
        for cx, cy, cw, ch in connected_regions:
            new_x = min(new_x, cx)
            new_y = min(new_y, cy)
            new_w = max(new_w, cx + cw - new_x)
            new_h = max(new_h, cy + ch - new_y)
        
        # Ensure boundaries don't exceed page limits
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(width - new_x, new_w)
        new_h = min(height - new_y, new_h)
        
        # Validate expanded region
        new_area = new_w * new_h
        if new_area > page_area * 0.6:  # Don't expand too much
            return None
        
        # Check if expansion is reasonable
        expansion_ratio = new_area / (w * h)
        if expansion_ratio > 3.0:  # Don't expand more than 3x
            return None
        
        return (new_x, new_y, new_w, new_h)
    
    def _detect_text_regions_around_figure(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> List[tuple]:
        """Detect text regions around the figure that might be captions or labels"""
        text_regions = []
        
        # Define search area around the figure
        search_margin = max(w, h) // 2
        search_x = max(0, x - search_margin)
        search_y = max(0, y - search_margin)
        search_w = min(gray.shape[1] - search_x, w + 2 * search_margin)
        search_h = min(gray.shape[0] - search_y, h + 2 * search_margin)
        
        # Extract search region
        search_region = gray[search_y:search_y+search_h, search_x:search_x+search_w]
        
        # Use morphological operations to detect text-like patterns
        # Horizontal lines (text baselines)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(search_region, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find contours in horizontal lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            tx, ty, tw, th = cv2.boundingRect(contour)
            if tw > 50 and th > 5:  # Minimum text size
                # Convert back to page coordinates
                tx += search_x
                ty += search_y
                text_regions.append((tx, ty, tw, th))
        
        return text_regions
    
    def _is_text_region_near_figure(self, tx: int, ty: int, tw: int, th: int, 
                                   fx: int, fy: int, fw: int, fh: int) -> bool:
        """Check if text region is close to the figure"""
        # Calculate centers
        text_center_x = tx + tw // 2
        text_center_y = ty + th // 2
        figure_center_x = fx + fw // 2
        figure_center_y = fy + fh // 2
        
        # Calculate distance
        distance = ((text_center_x - figure_center_x) ** 2 + 
                   (text_center_y - figure_center_y) ** 2) ** 0.5
        
        # Check if text is within reasonable distance
        max_distance = max(fw, fh) * 1.5
        return distance < max_distance
    
    def _find_connected_figure_components(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> List[tuple]:
        """Find connected components that might be part of the figure"""
        connected_regions = []
        
        # Define search area
        search_margin = max(w, h) // 3
        search_x = max(0, x - search_margin)
        search_y = max(0, y - search_margin)
        search_w = min(gray.shape[1] - search_x, w + 2 * search_margin)
        search_h = min(gray.shape[0] - search_y, h + 2 * search_margin)
        
        # Extract search region
        search_region = gray[search_y:search_y+search_h, search_x:search_x+search_w]
        
        # Threshold to create binary image
        _, binary = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        for i in range(1, num_labels):  # Skip background
            cx, cy, cw, ch, area = stats[i]
            
            # Check if component is reasonably sized and close to figure
            if (area > 100 and area < w * h * 2 and  # Not too small or too large
                self._is_text_region_near_figure(cx, cy, cw, ch, x-search_x, y-search_y, w, h)):
                
                # Convert back to page coordinates
                cx += search_x
                cy += search_y
                connected_regions.append((cx, cy, cw, ch))
        
        return connected_regions
    
    async def _detect_figures_by_templates(self, img: np.ndarray, gray: np.ndarray, 
                                   page_num: int, page) -> List[FigureCandidate]:
        """Detect figures using template patterns common in academic papers"""
        candidates = []
        
        # CV Chart detection disabled - other methods are sufficient
        # axis_candidates = await self._detect_chart_patterns(img, gray, page_num, page)
        # candidates.extend(axis_candidates)
        
        # Template 2: Diagram patterns (geometric shapes)
        diagram_candidates = self._detect_diagram_patterns(img, gray, page_num, page)
        candidates.extend(diagram_candidates)
        
        return candidates
    
    async def _detect_chart_patterns(self, img: np.ndarray, gray: np.ndarray, 
                             page_num: int, page) -> List[FigureCandidate]:
        """Detect chart/graph patterns with improved algorithm to avoid fragmentation"""
        candidates = []
        
        # Look for perpendicular lines (axes)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find intersections (potential chart origins)
        intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
        
        # Find regions around intersections
        contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Group nearby intersections to avoid fragmentation
        chart_regions = self._group_chart_regions(contours, img.shape)
        
        for idx, (x, y, w, h) in enumerate(chart_regions):
            area = w * h
            
            # Filter out very small fragments and ensure reasonable size
            min_chart_area = 5000  # Minimum area for a chart (smaller than MIN_FIGURE_AREA)
            
            if (min_chart_area < area < self.MAX_FIGURE_AREA and
                0.3 < w/h < 4.0):  # More flexible chart aspect ratio
                
                roi = img[y:y+h, x:x+w]
                
                if self._validate_chart_region(roi):
                    # Skip CV chart images for now (they contain more text content than figures)
                    # TODO: Improve chart detection to filter out text-heavy regions
                    image_path = None
                    if settings.store_locally:
                        # Store locally in organized folder structure for analysis
                        cv_chart_dir = self.output_dir / "figures" / "cv_chart"
                        cv_chart_dir.mkdir(parents=True, exist_ok=True)
                        img_path = cv_chart_dir / f"page{page_num}_fig{idx}.png"
                        cv2.imwrite(str(img_path), roi)
                        image_path = str(img_path)
                        logger.debug(f"Stored CV chart image locally (for analysis): {image_path}")
                    else:
                        # Skip upload for now
                        logger.debug(f"Skipping CV chart image upload (text-heavy): page{page_num}_fig{idx}")
                    
                    scale = page.rect.width / img.shape[1]
                    
                    candidate = FigureCandidate(
                        bbox=BoundingBox(
                            x1=x * scale,
                            y1=y * scale,
                            x2=(x + w) * scale,
                            y2=(y + h) * scale,
                            page=page_num,
                            confidence=0.75
                        ),
                        confidence=0.75,
                        method='cv_chart',
                        image_path=image_path,
                        label=f"Chart {page_num}.{idx}",
                        page=page_num
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _group_chart_regions(self, contours: List, img_shape: tuple) -> List[tuple]:
        """Group nearby chart intersections to avoid fragmentation"""
        if not contours:
            return []
        
        # Get bounding boxes for all contours
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, w, h))
        
        # Sort by area (largest first) to prioritize bigger regions
        regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        
        # Group overlapping or nearby regions
        merged_regions = []
        used = set()
        
        for i, (x1, y1, w1, h1) in enumerate(regions):
            if i in used:
                continue
                
            # Start with current region
            merged_x, merged_y = x1, y1
            merged_w, merged_h = w1, h1
            used.add(i)
            
            # Look for nearby regions to merge
            for j, (x2, y2, w2, h2) in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if regions are close enough to merge
                distance_threshold = 100  # pixels
                
                # Calculate center points
                cx1, cy1 = x1 + w1//2, y1 + h1//2
                cx2, cy2 = x2 + w2//2, y2 + h2//2
                
                # Calculate distance between centers
                distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                
                # Check for overlap or proximity
                overlap = (x1 < x2 + w2 and x2 < x1 + w1 and 
                          y1 < y2 + h2 and y2 < y1 + h1)
                
                if overlap or distance < distance_threshold:
                    # Merge regions
                    merged_x = min(merged_x, x2)
                    merged_y = min(merged_y, y2)
                    merged_w = max(merged_x + merged_w, x2 + w2) - merged_x
                    merged_h = max(merged_y + merged_h, y2 + h2) - merged_y
                    used.add(j)
            
            # Add some padding to capture full chart
            padding = 50
            merged_x = max(0, merged_x - padding)
            merged_y = max(0, merged_y - padding)
            merged_w = min(img_shape[1] - merged_x, merged_w + 2 * padding)
            merged_h = min(img_shape[0] - merged_y, merged_h + 2 * padding)
            
            merged_regions.append((merged_x, merged_y, merged_w, merged_h))
        
        return merged_regions
    
    def _validate_chart_region(self, roi: np.ndarray) -> bool:
        """Validate if region contains chart-like content"""
        if roi.size == 0:
            return False
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Calculate various features to distinguish charts from text
        score = 0
        
        # 1. Edge density analysis
        edges = cv2.Canny(gray_roi, 30, 100)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Charts typically have moderate edge density (not too sparse, not too dense)
        if 0.03 <= edge_density <= 0.3:
            score += 1
        
        # 2. Line detection (charts have more lines than text)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        if lines is not None and len(lines) > 5:
            score += 1
        
        # 3. Text content analysis (charts should have less text than figures)
        text_likelihood = self._estimate_text_content(gray_roi)
        if text_likelihood < 0.6:  # Lower threshold for charts
            score += 1
        
        # 4. Variance analysis (charts have more visual variation than text)
        variance = np.var(gray_roi)
        if variance > 150:  # Charts have more variation
            score += 1
        
        # 5. Aspect ratio check (charts are often wider than tall)
        h, w = gray_roi.shape
        aspect_ratio = w / h
        if 0.5 <= aspect_ratio <= 3.0:  # Reasonable chart aspect ratio
            score += 1
        
        # 6. Color diversity (charts often have multiple colors)
        if len(roi.shape) == 3:
            unique_colors = len(np.unique(roi.reshape(-1, roi.shape[2]), axis=0))
            if unique_colors > 30:  # Charts have more colors
                score += 1
        
        # Require at least 4 positive indicators for a chart
        return score >= 4
    
    def _detect_diagram_patterns(self, img: np.ndarray, gray: np.ndarray, 
                               page_num: int, page) -> List[FigureCandidate]:
        """Detect diagram patterns with geometric shapes"""
        # This method can be expanded based on specific diagram types
        # For now, return empty list
        return []
    
    async def _detect_figures_by_components(self, img: np.ndarray, gray: np.ndarray, 
                                    page_num: int, page) -> List[FigureCandidate]:
        """Detect figures using connected component analysis"""
        candidates = []
        
        # Threshold to create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            
            if (self.MIN_FIGURE_AREA < area < self.MAX_FIGURE_AREA and
                self.MIN_ASPECT_RATIO < w/h < self.MAX_ASPECT_RATIO):
                
                # Expand boundary to capture full figure including captions and labels
                page_area = img.shape[0] * img.shape[1]
                expanded_bbox = self._expand_figure_boundary(img, gray, x, y, w, h, page_area)
                if expanded_bbox:
                    x, y, w, h = expanded_bbox
                
                # Create mask for this component
                component_mask = (labels == i).astype(np.uint8) * 255
                
                # Extract region
                roi = img[y:y+h, x:x+w]
                
                if self._validate_component_region(roi, component_mask[y:y+h, x:x+w]):
                    # CV component images are decent quality - upload them
                    image_path = None
                    if settings.store_locally:
                        # Store locally in organized folder structure
                        cv_component_dir = self.output_dir / "figures" / "cv_component"
                        cv_component_dir.mkdir(parents=True, exist_ok=True)
                        img_path = cv_component_dir / f"page{page_num}_fig{i}.png"
                        cv2.imwrite(str(img_path), roi)
                        image_path = str(img_path)
                        logger.debug(f"Stored CV component image locally: {image_path}")
                    else:
                        # Upload to Cloudinary
                        filename = f"cv_component_page{page_num}_fig{i}"
                        image_path = await cloudinary_service.upload_cv_image(roi, filename, "scholarai/figures")
                        logger.debug(f"Uploaded CV component image to Cloudinary: {image_path}")
                    
                    scale = page.rect.width / img.shape[1]
                    
                    candidate = FigureCandidate(
                        bbox=BoundingBox(
                            x1=x * scale,
                            y1=y * scale,
                            x2=(x + w) * scale,
                            y2=(y + h) * scale,
                            page=page_num,
                            confidence=0.65
                        ),
                        confidence=0.65,
                        method='cv_component',
                        image_path=image_path,
                        label=f"Component {page_num}.{i}",
                        page=page_num
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _validate_component_region(self, roi: np.ndarray, mask: np.ndarray) -> bool:
        """Validate connected component as potential figure"""
        if roi.size == 0 or mask.size == 0:
            return False
        
        # Check component density
        mask_ratio = np.count_nonzero(mask) / mask.size
        
        # Good components have moderate density (not too sparse or dense)
        return 0.1 <= mask_ratio <= 0.7
    
    def _validate_figure_region(self, roi: np.ndarray) -> bool:
        """General validation for figure regions with improved tolerance for expanded boundaries"""
        if roi.size == 0:
            return False
        
        # For expanded boundaries, be more lenient with validation
        # since they may include captions and labels
        return self._is_likely_figure_content_expanded(roi)
    
    def _is_likely_figure_content_expanded(self, img: np.ndarray) -> bool:
        """Enhanced validation for expanded figure regions that may include captions"""
        if img is None or img.size == 0:
            return False
        
        height, width = img.shape[:2]
        
        # More lenient size validation for expanded regions
        area = width * height
        if area < self.MIN_FIGURE_AREA * 0.5 or area > self.MAX_FIGURE_AREA * 1.5:
            return False
        
        # More flexible aspect ratio for expanded regions
        aspect_ratio = width / height
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Much more flexible
            return False
        
        # Convert to grayscale for analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Content analysis with lower thresholds
        score = 0
        
        # 1. Edge density (more lenient for expanded regions)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        if 0.01 <= edge_ratio <= 0.4:  # More lenient range
            score += 1
        
        # 2. Variance (figures have variation, not uniform backgrounds)
        variance = np.var(gray)
        if variance > 100:  # Lower threshold
            score += 1
        
        # 3. Color diversity (figures often have multiple colors)
        if len(img.shape) == 3:
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            if unique_colors > 20:  # Lower threshold
                score += 1
        
        # 4. Text content analysis (expanded regions may have more text)
        text_likelihood = self._estimate_text_content(gray)
        if text_likelihood < 0.8:  # Higher threshold (more text allowed)
            score += 1
        
        # 5. Geometric shapes detection
        if self._detect_geometric_patterns(gray):
            score += 1
        
        # Require at least 2 positive indicators (lower threshold)
        return score >= 2

    # Caption detection and validation methods
    async def _detect_captions(self, candidates: List[FigureCandidate], 
                             pdf_path: Path) -> List[FigureCandidate]:
        """Enhanced caption detection for figure candidates"""
        doc = fitz.open(str(pdf_path))
        
        for candidate in candidates:
            try:
                page = doc[candidate.page - 1]
                
                # Get text around figure
                nearby_text = self._get_text_near_figure(page, candidate.bbox)
                
                # Extract caption using patterns
                caption, label = self._extract_caption_from_text(nearby_text)
                
                if caption:
                    candidate.caption = caption
                if label and not candidate.label:
                    candidate.label = label
                    
            except Exception as e:
                logger.warning(f"Caption detection failed for candidate on page {candidate.page}: {e}")
        
        doc.close()
        return candidates
    
    def _get_text_near_figure(self, page, figure_bbox: BoundingBox, 
                            proximity: float = 100) -> List[Dict[str, Any]]:
        """Get text blocks near figure"""
        text_dict = page.get_text("dict")
        nearby_text = []
        
        for block in text_dict["blocks"]:
            if block.get("type") == 0:  # Text block
                block_bbox = block["bbox"]
                
                # Check if block is near figure
                if self._is_text_near_figure(block_bbox, figure_bbox, proximity):
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                    
                    nearby_text.append({
                        'text': block_text.strip(),
                        'bbox': block_bbox,
                        'distance': self._calculate_bbox_distance(block_bbox, figure_bbox)
                    })
        
        # Sort by distance to figure
        nearby_text.sort(key=lambda x: x['distance'])
        return nearby_text
    
    def _is_text_near_figure(self, text_bbox: List[float], figure_bbox: BoundingBox, 
                           proximity: float) -> bool:
        """Check if text is near figure"""
        distance = self._calculate_bbox_distance(text_bbox, figure_bbox)
        return distance <= proximity
    
    def _calculate_bbox_distance(self, text_bbox: List[float], 
                               figure_bbox: BoundingBox) -> float:
        """Calculate distance between bounding boxes"""
        # Calculate center points
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2
        text_center_y = (text_bbox[1] + text_bbox[3]) / 2
        
        fig_center_x = (figure_bbox.x1 + figure_bbox.x2) / 2
        fig_center_y = (figure_bbox.y1 + figure_bbox.y2) / 2
        
        # Euclidean distance
        return np.sqrt((text_center_x - fig_center_x)**2 + 
                      (text_center_y - fig_center_y)**2)
    
    def _extract_caption_from_text(self, nearby_text: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """Extract caption and label from nearby text"""
        caption = None
        label = None
        
        for text_block in nearby_text:
            text = text_block['text']
            
            for pattern in self.CAPTION_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    if len(match.groups()) >= 2:
                        figure_num = match.group(1)
                        caption_text = match.group(2).strip()
                        
                        # Clean caption text
                        caption_text = re.sub(r'\s+', ' ', caption_text)
                        caption_text = caption_text.strip('.,;:')
                        
                        if caption_text and len(caption_text) > 5:  # Reasonable caption length
                            label = f"Figure {figure_num}"
                            caption = caption_text
                            return caption, label
        
        return caption, label
    
    async def _validate_figure_candidates(self, candidates: List[FigureCandidate], 
                                        pdf_path: Path) -> List[FigureCandidate]:
        """Validate figure candidates using multiple criteria"""
        validated = []
        
        for candidate in candidates:
            validation_score = await self._calculate_figure_validation_score(candidate, pdf_path)
            
            if validation_score > 0.6:  # Validation threshold
                candidate.confidence = (candidate.confidence + validation_score) / 2
                validated.append(candidate)
                logger.debug(f"Figure validated: {candidate.label}, score: {validation_score:.2f}")
            else:
                logger.debug(f"Figure rejected: {candidate.label}, score: {validation_score:.2f}")
        
        return validated
    
    async def _calculate_figure_validation_score(self, candidate: FigureCandidate, 
                                               pdf_path: Path) -> float:
        """Calculate comprehensive validation score for figure"""
        score = 0.0
        
        # 1. Size and proportion validation (0.2 weight)
        bbox_score = self._validate_figure_bbox(candidate.bbox)
        score += 0.2 * bbox_score
        
        # 2. Caption quality (0.3 weight)
        caption_score = self._validate_caption_quality(candidate.caption, candidate.label)
        score += 0.3 * caption_score
        
        # 3. Image content validation (0.3 weight)
        if candidate.image_path and Path(candidate.image_path).exists():
            content_score = self._validate_image_content(candidate.image_path)
            score += 0.3 * content_score
        else:
            score += 0.15  # Partial score if no image available
        
        # 4. Method confidence (0.2 weight)
        method_score = self._get_method_confidence_score(candidate.method)
        score += 0.2 * method_score
        
        return min(1.0, score)
    
    async def _extract_ocr_from_figure(self, candidate: FigureCandidate, pdf_path: Path) -> FigureCandidate:
        """Extract OCR text from figure image for LLM processing"""
        if not self._is_ocr_available() or not candidate.image_path:
            return candidate
        
        try:
            # Load the figure image
            image_path = Path(candidate.image_path)
            if not image_path.exists():
                return candidate
            
            # Extract text using OCR manager
            ocr_result = await self.ocr_manager.extract_text(image_path)
            
            if ocr_result and ocr_result.text.strip():
                candidate.ocr_text = ocr_result.text
                candidate.ocr_confidence = ocr_result.confidence
                
                logger.debug(f"OCR extracted from figure {candidate.label}: {len(ocr_result.text)} chars, confidence: {ocr_result.confidence:.2f}, provider: {ocr_result.provider}")
            
        except Exception as e:
            logger.warning(f"OCR extraction failed for figure {candidate.label}: {e}")
        
        return candidate
    

    
    async def _extract_ocr_from_pdf_region(self, candidate: FigureCandidate, pdf_path: Path) -> FigureCandidate:
        """Extract OCR text from PDF region when image file is not available"""
        if not self._is_ocr_available():
            return candidate
        
        try:
            # Open PDF and extract the region as image
            doc = fitz.open(str(pdf_path))
            page = doc[candidate.page - 1]  # Convert to 0-based index
            
            # Create transformation matrix for the figure region
            bbox = candidate.bbox
            rect = fitz.Rect(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            
            # Extract region as image with higher resolution
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat, clip=rect)
            
            # Convert to bytes
            img_data = pix.tobytes("png")
            
            # Extract text using OCR manager
            ocr_result = await self.ocr_manager.extract_text_from_bytes(img_data)
            
            if ocr_result and ocr_result.text.strip():
                candidate.ocr_text = ocr_result.text
                candidate.ocr_confidence = ocr_result.confidence
                
                logger.debug(f"OCR extracted from PDF region {candidate.label}: {len(ocr_result.text)} chars, confidence: {ocr_result.confidence:.2f}, provider: {ocr_result.provider}")
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"PDF region OCR extraction failed for figure {candidate.label}: {e}")
        
        return candidate
    
    def _validate_figure_bbox(self, bbox: BoundingBox) -> float:
        """Validate figure bounding box dimensions"""
        width = bbox.x2 - bbox.x1
        height = bbox.y2 - bbox.y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        score = 0.0
        
        # Size validation
        if self.MIN_FIGURE_AREA <= area <= self.MAX_FIGURE_AREA:
            score += 0.5
        
        # Aspect ratio validation
        if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
            score += 0.5
        
        return score
    
    def _validate_caption_quality(self, caption: Optional[str], 
                                label: Optional[str]) -> float:
        """Validate quality of detected caption"""
        score = 0.0
        
        # Label validation
        if label and re.match(r'(?:Figure|Fig\.?)\s*\d+', label, re.IGNORECASE):
            score += 0.4
        
        # Caption validation
        if caption:
            # Length check
            if 10 <= len(caption) <= 500:  # Reasonable caption length
                score += 0.3
            
            # Content check (should be descriptive)
            descriptive_words = ['shows', 'displays', 'presents', 'illustrates', 
                               'depicts', 'comparison', 'results', 'data']
            if any(word in caption.lower() for word in descriptive_words):
                score += 0.2
            
            # Sentence structure
            if caption.endswith('.') and len(caption.split()) >= 3:
                score += 0.1
        
        return score
    
    def _validate_image_content(self, image_path: str) -> float:
        """Validate actual image content"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            # Reuse the figure content validation
            return 1.0 if self._is_likely_figure_content(img) else 0.3
            
        except Exception as e:
            logger.warning(f"Image validation failed for {image_path}: {e}")
            return 0.5  # Neutral score if validation fails
    
    def _get_method_confidence_score(self, method: str) -> float:
        """Get confidence score based on extraction method"""
        method_scores = {
            'pdffigures2': 0.9,
            'cv_contour': 0.7,
            'cv_chart': 0.75,
            'cv_component': 0.6
        }
        return method_scores.get(method, 0.5)
    
    async def _match_text_references(self, candidates: List[FigureCandidate], 
                                   pdf_path: Path) -> List[FigureCandidate]:
        """Match figures with text references"""
        # Find all figure references in document
        doc = fitz.open(str(pdf_path))
        all_references = {}
        
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            
            # Find figure references
            references = re.findall(r'(?:Figure|Fig\.?)\s*(\d+)', page_text, re.IGNORECASE)
            for ref in references:
                if ref not in all_references:
                    all_references[ref] = []
                all_references[ref].append(page_num)
        
        doc.close()
        
        # Match candidates with references
        for candidate in candidates:
            if candidate.label:
                fig_num = re.search(r'\d+', candidate.label)
                if fig_num:
                    num = fig_num.group()
                    if num in all_references:
                        candidate.confidence += 0.1  # Boost confidence for referenced figures
                        # Store reference information
                        if not hasattr(candidate, 'references'):
                            candidate.references = []
                        # Convert references to strings
                        candidate.references.extend([str(ref) for ref in all_references[num]])
        
        return candidates
    
    def _deduplicate_and_rank_figures(self, candidates: List[FigureCandidate]) -> List[Figure]:
        """Deduplicate candidates and convert to final Figure objects"""
        # Group by page
        page_candidates = {}
        for candidate in candidates:
            if candidate.page not in page_candidates:
                page_candidates[candidate.page] = []
            page_candidates[candidate.page].append(candidate)
        
        final_figures = []
        
        for page_num, page_cands in page_candidates.items():
            # Remove duplicates within page
            unique_cands = self._remove_duplicate_candidates(page_cands)
            
            # Sort by confidence
            unique_cands.sort(key=lambda x: x.confidence, reverse=True)
            
            # Convert to Figure objects
            for idx, candidate in enumerate(unique_cands):
                figure = Figure(
                    label=candidate.label or f"Figure {page_num}.{idx}",
                    caption=candidate.caption,
                    page=candidate.page,
                    bbox=candidate.bbox,
                    image_path=candidate.image_path,
                    type=self._determine_figure_type(candidate),
                    references=getattr(candidate, 'references', []),
                    extraction_method=candidate.method,
                    confidence=candidate.confidence,
                    ocr_text=candidate.ocr_text,  # Include OCR extracted text
                    ocr_confidence=candidate.ocr_confidence  # Include OCR confidence
                )
                final_figures.append(figure)
        
        return final_figures
    
    def _remove_duplicate_candidates(self, candidates: List[FigureCandidate], 
                                   iou_threshold: float = 0.5) -> List[FigureCandidate]:
        """Remove duplicate candidates based on bbox overlap"""
        unique_candidates = []
        
        for candidate in candidates:
            is_duplicate = False
            
            for unique_cand in unique_candidates:
                iou = self._calculate_bbox_iou(candidate.bbox, unique_cand.bbox)
                if iou > iou_threshold:
                    # Keep the one with higher confidence
                    if candidate.confidence > unique_cand.confidence:
                        unique_candidates.remove(unique_cand)
                        unique_candidates.append(candidate)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _calculate_bbox_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate IoU between two bounding boxes"""
        # Calculate intersection
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_figure_type(self, candidate: FigureCandidate) -> str:
        """Determine figure type based on content and method"""
        if 'chart' in candidate.method:
            return 'chart'
        elif 'diagram' in candidate.method:
            return 'diagram'
        else:
            return 'figure'