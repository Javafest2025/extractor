# services/extractors/table_extractor.py
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from loguru import logger
import json
import io

from app.models.schemas import Table, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError


class TableExtractor:
    """
    Multi-method table extraction using:
    1. Table Transformer (deep learning)
    2. PDFPlumber (rule-based)
    3. PyMuPDF (structure-based)
    4. Computer vision (fallback)
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "tables"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        self._init_models()
    
    def _init_models(self):
        """Initialize deep learning models"""
        try:
            # Table detection model
            self.table_detector = AutoModelForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            ).to(self.device)
            self.table_detector.eval()
            
            # Table structure recognition model
            self.table_structure = AutoModelForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            ).to(self.device)
            self.table_structure.eval()
            
            # Image processor
            self.image_processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            
            self.dl_models_available = True
            logger.info("Table Transformer models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Table Transformer models: {e}")
            self.dl_models_available = False
    
    async def extract(self, pdf_path: Path) -> List[Table]:
        """Extract tables using multiple methods for maximum coverage"""
        all_tables = []
        
        # Method 1: PDFPlumber (fast, works well for simple tables)
        try:
            pdfplumber_tables = await self._extract_with_pdfplumber(pdf_path)
            all_tables.extend(pdfplumber_tables)
            logger.info(f"PDFPlumber extracted {len(pdfplumber_tables)} tables")
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
        
        # Method 2: Table Transformer (best for complex tables)
        if self.dl_models_available:
            try:
                transformer_tables = await self._extract_with_transformer(pdf_path)
                new_tables = self._deduplicate_tables(all_tables, transformer_tables)
                all_tables.extend(new_tables)
                logger.info(f"Table Transformer extracted {len(new_tables)} additional tables")
            except Exception as e:
                logger.error(f"Table Transformer extraction failed: {e}")
        
        # Method 3: PyMuPDF structure analysis
        try:
            pymupdf_tables = await self._extract_with_pymupdf(pdf_path)
            new_tables = self._deduplicate_tables(all_tables, pymupdf_tables)
            all_tables.extend(new_tables)
            logger.info(f"PyMuPDF extracted {len(new_tables)} additional tables")
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
        
        # Post-process: enhance table structure for all extracted tables
        for table in all_tables:
            if not table.headers or not table.rows:
                await self._enhance_table_structure(table, pdf_path)
        
        return all_tables
    
    async def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Table]:
        """Extract tables using PDFPlumber"""
        tables = []
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                
                for idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # Parse table structure
                    headers = table_data[0] if table_data else []
                    rows = table_data[1:] if len(table_data) > 1 else []
                    
                    # Clean data
                    headers = [str(h) if h else "" for h in headers]
                    rows = [[str(cell) if cell else "" for cell in row] for row in rows]
                    
                    # Normalize column count - ensure all rows have same number of columns
                    if headers:
                        max_cols = len(headers)
                    else:
                        max_cols = max(len(row) for row in rows) if rows else 0
                    
                    # Pad headers if needed
                    while len(headers) < max_cols:
                        headers.append(f"Column {len(headers)}")
                    
                    # Pad rows if needed
                    normalized_rows = []
                    for row in rows:
                        normalized_row = row[:]
                        while len(normalized_row) < max_cols:
                            normalized_row.append("")
                        # Truncate if too long
                        normalized_row = normalized_row[:max_cols]
                        normalized_rows.append(normalized_row)
                    
                    # Get table bbox (approximate from page)
                    # PDFPlumber doesn't provide exact bbox, so we estimate
                    bbox = BoundingBox(
                        x1=72,  # 1 inch margin
                        y1=72,
                        x2=page.width - 72,
                        y2=page.height - 72,
                        page=page_num
                    )
                    
                    # Save as CSV
                    csv_path = self.output_dir / f"page{page_num}_table{idx}.csv"
                    try:
                        df = pd.DataFrame(normalized_rows, columns=headers[:max_cols])
                        df.to_csv(csv_path, index=False)
                    except Exception as e:
                        logger.warning(f"Failed to save table as CSV: {e}")
                        # Fallback: save as raw CSV
                        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                            import csv
                            writer = csv.writer(f)
                            writer.writerow(headers[:max_cols])
                            writer.writerows(normalized_rows)
                    
                    # Create HTML representation
                    try:
                        html = df.to_html(index=False, classes='table table-bordered')
                    except:
                        html = None
                    
                    table = Table(
                        label=f"Table {page_num}.{idx}",
                        page=page_num,
                        bbox=bbox,
                        headers=[headers[:max_cols]],
                        rows=normalized_rows,
                        csv_path=str(csv_path),
                        html=html
                    )
                    tables.append(table)
        
        return tables
    
    async def _extract_with_transformer(self, pdf_path: Path) -> List[Table]:
        """Extract tables using Table Transformer deep learning models"""
        tables = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Detect tables
            table_regions = await self._detect_tables(pil_image)
            
            for idx, region in enumerate(table_regions):
                # Crop table region
                x1, y1, x2, y2 = region
                table_image = pil_image.crop((x1, y1, x2, y2))
                
                # Recognize table structure
                structure = await self._recognize_structure(table_image)
                
                # Extract text from cells
                headers, rows = await self._extract_cell_text(
                    page, region, structure
                )
                
                # Save table image
                img_path = self.output_dir / f"transformer_page{page_num}_table{idx}.png"
                table_image.save(img_path)
                
                # Save as CSV
                if headers and rows:
                    csv_path = self.output_dir / f"transformer_page{page_num}_table{idx}.csv"
                    df = pd.DataFrame(rows, columns=headers[0] if headers else None)
                    df.to_csv(csv_path, index=False)
                    html = df.to_html(index=False, classes='table table-bordered')
                else:
                    csv_path = None
                    html = None
                
                # Convert coordinates to PDF space
                scale = page.rect.width / pil_image.width
                
                table = Table(
                    label=f"Table {page_num}.{idx}",
                    page=page_num,
                    bbox=BoundingBox(
                        x1=x1 * scale,
                        y1=y1 * scale,
                        x2=x2 * scale,
                        y2=y2 * scale,
                        page=page_num,
                        confidence=0.9
                    ),
                    headers=headers,
                    rows=rows,
                    csv_path=str(csv_path) if csv_path else None,
                    html=html,
                    structure=structure
                )
                tables.append(table)
        
        doc.close()
        return tables
    
    async def _detect_tables(self, image: Image.Image) -> List[tuple]:
        """Detect table regions in an image using Table Transformer"""
        # Prepare image
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run detection
        with torch.no_grad():
            outputs = self.table_detector(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]
        
        # Extract bounding boxes
        tables = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            if score > 0.9:  # High confidence threshold
                box = [int(i) for i in box.tolist()]
                tables.append(tuple(box))
        
        return tables
    
    async def _recognize_structure(self, table_image: Image.Image) -> Dict[str, Any]:
        """Recognize table structure (rows, columns, headers)"""
        # Prepare image
        inputs = self.image_processor(images=table_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run structure recognition
        with torch.no_grad():
            outputs = self.table_structure(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([table_image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.7
        )[0]
        
        # Organize structure
        structure = {
            "columns": [],
            "rows": [],
            "headers": [],
            "cells": []
        }
        
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [int(i) for i in box.tolist()]
            label_name = self._get_label_name(label.item())
            
            if label_name == "table column":
                structure["columns"].append(box)
            elif label_name == "table row":
                structure["rows"].append(box)
            elif label_name == "table column header":
                structure["headers"].append(box)
            elif label_name == "table cell":
                structure["cells"].append(box)
        
        # Sort by position
        structure["columns"].sort(key=lambda x: x[0])
        structure["rows"].sort(key=lambda x: x[1])
        structure["headers"].sort(key=lambda x: x[0])
        
        return structure
    
    async def _extract_with_pymupdf(self, pdf_path: Path) -> List[Table]:
        """Extract tables using PyMuPDF's structure analysis"""
        tables = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Get page text with structure
            blocks = page.get_text("dict")
            
            # Find potential table blocks
            table_blocks = self._identify_table_blocks(blocks)
            
            for idx, table_block in enumerate(table_blocks):
                # Extract table data
                headers, rows = self._parse_table_block(table_block)
                
                if not headers and not rows:
                    continue
                
                # Calculate bounding box
                bbox = self._calculate_block_bbox(table_block)
                
                # Save as CSV
                csv_path = self.output_dir / f"mupdf_page{page_num}_table{idx}.csv"
                if headers and rows:
                    try:
                        # Normalize column count
                        header_row = headers[0] if headers else []
                        max_cols = max(len(header_row), max(len(row) for row in rows) if rows else 0)
                        
                        # Pad headers and rows
                        normalized_header = header_row[:]
                        while len(normalized_header) < max_cols:
                            normalized_header.append(f"Column {len(normalized_header)}")
                        
                        normalized_rows = []
                        for row in rows:
                            normalized_row = row[:]
                            while len(normalized_row) < max_cols:
                                normalized_row.append("")
                            normalized_row = normalized_row[:max_cols]
                            normalized_rows.append(normalized_row)
                        
                        df = pd.DataFrame(normalized_rows, columns=normalized_header[:max_cols])
                        df.to_csv(csv_path, index=False)
                        html = df.to_html(index=False, classes='table table-bordered')
                        headers = [normalized_header[:max_cols]]
                        rows = normalized_rows
                    except Exception as e:
                        logger.warning(f"Failed to normalize PyMuPDF table: {e}")
                        csv_path = None
                        html = None
                else:
                    csv_path = None
                    html = None
                
                table = Table(
                    label=f"Table {page_num}.{idx}",
                    page=page_num,
                    bbox=BoundingBox(
                        x1=bbox[0],
                        y1=bbox[1],
                        x2=bbox[2],
                        y2=bbox[3],
                        page=page_num,
                        confidence=0.7
                    ),
                    headers=headers,
                    rows=rows,
                    csv_path=str(csv_path) if csv_path else None,
                    html=html
                )
                tables.append(table)
        
        doc.close()
        return tables
    
    def _identify_table_blocks(self, blocks: Dict) -> List[Dict]:
        """Identify blocks that likely contain tables"""
        table_blocks = []
        
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # Text block
                lines = block.get("lines", [])
                
                # Heuristics for table detection
                if self._is_likely_table(lines):
                    table_blocks.append(block)
        
        return table_blocks
    
    def _is_likely_table(self, lines: List[Dict]) -> bool:
        """Check if lines likely form a table"""
        if len(lines) < 2:
            return False
        
        # Check for consistent column alignment
        x_positions = []
        for line in lines:
            for span in line.get("spans", []):
                x_positions.append(span.get("bbox", [0])[0])
        
        if not x_positions:
            return False
        
        # Count unique x positions (potential columns)
        unique_x = len(set(round(x, 1) for x in x_positions))
        
        # Tables typically have multiple columns
        return unique_x >= 2 and len(lines) >= 3
    
    def _parse_table_block(self, block: Dict) -> tuple:
        """Parse table structure from a text block"""
        lines = block.get("lines", [])
        if not lines:
            return [], []
        
        # Group text by lines and x-position
        rows_data = []
        for line in lines:
            row = []
            for span in sorted(line.get("spans", []), key=lambda x: x.get("bbox", [0])[0]):
                row.append(span.get("text", "").strip())
            if row:
                rows_data.append(row)
        
        if not rows_data:
            return [], []
        
        # Assume first row is header
        headers = [rows_data[0]] if rows_data else []
        rows = rows_data[1:] if len(rows_data) > 1 else []
        
        return headers, rows
    
    def _calculate_block_bbox(self, block: Dict) -> tuple:
        """Calculate bounding box for a block"""
        bbox = block.get("bbox", [0, 0, 100, 100])
        return tuple(bbox)
    
    async def _extract_cell_text(self, page, region: tuple, structure: Dict) -> tuple:
        """Extract text from table cells based on structure"""
        headers = []
        rows = []
        
        # Get text from page
        page_dict = page.get_text("dict")
        
        # Extract header text
        if structure["headers"]:
            header_row = []
            for header_bbox in sorted(structure["headers"], key=lambda x: x[0]):
                text = self._get_text_in_region(page_dict, header_bbox)
                header_row.append(text)
            headers.append(header_row)
        
        # Extract row text
        if structure["rows"] and structure["columns"]:
            for row_bbox in structure["rows"]:
                row_data = []
                for col_bbox in structure["columns"]:
                    # Find intersection of row and column
                    cell_bbox = self._intersect_boxes(row_bbox, col_bbox)
                    if cell_bbox:
                        text = self._get_text_in_region(page_dict, cell_bbox)
                        row_data.append(text)
                if row_data:
                    rows.append(row_data)
        
        return headers, rows
    
    def _get_text_in_region(self, page_dict: Dict, bbox: tuple) -> str:
        """Extract text within a bounding box"""
        x1, y1, x2, y2 = bbox
        texts = []
        
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_bbox = span.get("bbox", [])
                        if len(span_bbox) == 4:
                            sx1, sy1, sx2, sy2 = span_bbox
                            # Check if span is within region
                            if sx1 >= x1 and sy1 >= y1 and sx2 <= x2 and sy2 <= y2:
                                texts.append(span.get("text", ""))
        
        return " ".join(texts).strip()
    
    def _intersect_boxes(self, box1: tuple, box2: tuple) -> Optional[tuple]:
        """Find intersection of two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
        return None
    
    def _deduplicate_tables(self,
                          existing: List[Table],
                          new: List[Table],
                          iou_threshold: float = 0.5) -> List[Table]:
        """Remove duplicate tables based on page and bbox overlap"""
        unique_new = []
        
        for new_table in new:
            is_duplicate = False
            
            for exist_table in existing:
                if new_table.page == exist_table.page:
                    iou = self._calculate_iou(new_table.bbox, exist_table.bbox)
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_new.append(new_table)
        
        return unique_new
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_label_name(self, label_id: int) -> str:
        """Map label ID to name for Table Transformer"""
        label_map = {
            0: "table",
            1: "table column",
            2: "table row",
            3: "table column header",
            4: "table cell",
            5: "table spanning cell",
            6: "no object"
        }
        return label_map.get(label_id, "unknown")
    
    async def _enhance_table_structure(self, table: Table, pdf_path: Path):
        """Enhance table structure using OCR if needed"""
        if not table.image_path:
            return
        
        # Load table image
        img = cv2.imread(table.image_path)
        if img is None:
            return
        
        # Apply OCR to extract text
        import pytesseract
        text = pytesseract.image_to_string(img)
        
        # Parse text into rows
        lines = text.strip().split('\n')
        if lines:
            # Simple parsing - can be enhanced
            rows = []
            for line in lines:
                # Split by multiple spaces or tabs
                import re
                cells = re.split(r'\s{2,}|\t', line.strip())
                if cells:
                    rows.append(cells)
            
            if rows and not table.rows:
                table.headers = [rows[0]] if rows else []
                table.rows = rows[1:] if len(rows) > 1 else []

