# services/extractors/figure_extractor.py
import subprocess
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from loguru import logger
import httpx

from app.models.schemas import Figure, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError


class FigureExtractor:
    """
    Multi-method figure extraction using PDFFigures2 as primary
    and computer vision techniques as fallback
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "figures"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.pdffigures2_available = self._check_pdffigures2()
        
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
    
    async def extract(self, pdf_path: Path) -> List[Figure]:
        """
        Extract figures using multiple methods for maximum coverage
        """
        figures = []
        
        # Method 1: PDFFigures2 (best for academic papers)
        if self.pdffigures2_available:
            try:
                pdffigures_results = await self._extract_with_pdffigures2(pdf_path)
                figures.extend(pdffigures_results)
                logger.info(f"PDFFigures2 extracted {len(pdffigures_results)} figures")
            except Exception as e:
                logger.error(f"PDFFigures2 extraction failed: {e}")
        
        # Method 2: PyMuPDF image extraction (for embedded images)
        try:
            pymupdf_results = await self._extract_with_pymupdf(pdf_path)
            # Deduplicate based on page and approximate position
            new_figures = self._deduplicate_figures(figures, pymupdf_results)
            figures.extend(new_figures)
            logger.info(f"PyMuPDF extracted {len(new_figures)} additional figures")
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
        
        # Method 3: Computer vision detection (for complex layouts)
        try:
            cv_results = await self._extract_with_cv(pdf_path)
            new_figures = self._deduplicate_figures(figures, cv_results)
            figures.extend(new_figures)
            logger.info(f"CV detection found {len(new_figures)} additional figures")
        except Exception as e:
            logger.error(f"CV detection failed: {e}")
        
        return figures
    
    async def _extract_with_pdffigures2(self, pdf_path: Path) -> List[Figure]:
        """Extract figures using PDFFigures2"""
        output_prefix = self.output_dir / pdf_path.stem
        
        # Run PDFFigures2
        if settings.pdffigures2_path.startswith("docker"):
            # Docker command
            cmd = settings.pdffigures2_path.split() + [
                str(pdf_path),
                "-m", str(output_prefix),  # save images
                "-d", str(self.output_dir),  # output directory
                "-j", str(output_prefix) + ".json"  # JSON output
            ]
        else:
            # Local binary
            cmd = [
                settings.pdffigures2_path,
                str(pdf_path),
                "-m", str(output_prefix),  # save images
                "-d", str(self.output_dir),  # output directory
                "-j", str(output_prefix) + ".json"  # JSON output
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
        
        figures = []
        for fig_data in data:
            # Create Figure object
            figure = Figure(
                label=fig_data.get('name', ''),
                caption=fig_data.get('caption', ''),
                page=fig_data.get('page', 0),
                bbox=BoundingBox(
                    x1=fig_data.get('regionBoundary', {}).get('x1', 0),
                    y1=fig_data.get('regionBoundary', {}).get('y1', 0),
                    x2=fig_data.get('regionBoundary', {}).get('x2', 0),
                    y2=fig_data.get('regionBoundary', {}).get('y2', 0),
                    page=fig_data.get('page', 0)
                ),
                type=fig_data.get('figType', 'figure'),
                image_path=fig_data.get('renderURL', '')
            )
            figures.append(figure)
        
        return figures
    
    async def _extract_with_pymupdf(self, pdf_path: Path) -> List[Figure]:
        """Extract embedded images using PyMuPDF"""
        figures = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Get images on this page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix.tobytes("png")
                    
                    # Save image
                    img_path = self.output_dir / f"page{page_num}_img{img_index}.png"
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    
                    # Get image position on page
                    img_rect = page.get_image_bbox(img)
                    
                    # Create Figure object
                    figure = Figure(
                        label=f"Figure {page_num}.{img_index}",
                        page=page_num,
                        bbox=BoundingBox(
                            x1=img_rect.x0,
                            y1=img_rect.y0,
                            x2=img_rect.x1,
                            y2=img_rect.y1,
                            page=page_num,
                            confidence=0.9
                        ),
                        image_path=str(img_path),
                        type="embedded_image"
                    )
                    figures.append(figure)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
        
        doc.close()
        return figures
    
    async def _extract_with_cv(self, pdf_path: Path) -> List[Figure]:
        """
        Use computer vision to detect figures in rendered PDF pages
        Useful for detecting charts, diagrams, and complex figures
        """
        figures = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect potential figure regions using edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and process contours
            height, width = img.shape[:2]
            min_area = (width * height) * 0.01  # Minimum 1% of page area
            max_area = (width * height) * 0.5   # Maximum 50% of page area
            
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (filter out text blocks)
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 3:  # Reasonable figure aspect ratio
                        # Extract region
                        roi = img[y:y+h, x:x+w]
                        
                        # Check if region has enough variation (not just text)
                        if self._is_likely_figure(roi):
                            # Save extracted figure
                            fig_path = self.output_dir / f"cv_page{page_num}_fig{idx}.png"
                            cv2.imwrite(str(fig_path), roi)
                            
                            # Convert coordinates back to PDF space
                            scale = page.rect.width / width
                            
                            figure = Figure(
                                label=f"Figure {page_num}.{idx}",
                                page=page_num,
                                bbox=BoundingBox(
                                    x1=x * scale,
                                    y1=y * scale,
                                    x2=(x + w) * scale,
                                    y2=(y + h) * scale,
                                    page=page_num,
                                    confidence=0.7
                                ),
                                image_path=str(fig_path),
                                type="cv_detected"
                            )
                            figures.append(figure)
        
        doc.close()
        return figures
    
    def _is_likely_figure(self, img: np.ndarray) -> bool:
        """
        Heuristic to determine if an image region is likely a figure
        (not just text or whitespace)
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Check variance (figures have more variation than text blocks)
        variance = np.var(gray)
        if variance < 100:
            return False
        
        # Check for presence of lines (common in charts/diagrams)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        
        # Check color distribution (figures often have multiple colors)
        if len(img.shape) == 3:
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            if unique_colors > 100:  # More than 100 unique colors
                return True
        
        # Combine heuristics
        return edge_ratio > 0.05  # At least 5% edges
    
    def _deduplicate_figures(self, 
                           existing: List[Figure], 
                           new: List[Figure],
                           iou_threshold: float = 0.5) -> List[Figure]:
        """
        Remove duplicate figures based on page and bounding box overlap
        """
        unique_new = []
        
        for new_fig in new:
            is_duplicate = False
            
            for exist_fig in existing:
                if new_fig.page == exist_fig.page:
                    # Calculate IoU (Intersection over Union)
                    iou = self._calculate_iou(new_fig.bbox, exist_fig.bbox)
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_new.append(new_fig)
        
        return unique_new
    
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