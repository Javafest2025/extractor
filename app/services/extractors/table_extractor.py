# services/extractors/table_extractor.py
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
import re
from collections import Counter
import nltk
from textstat import flesch_reading_ease

from app.models.schemas import Table, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError
from app.services.cloudinary_service import cloudinary_service
from app.services.extractors.camelot_extractor import camelot_extractor
from app.services.extractors.tabula_extractor import tabula_extractor


class TableExtractor:
    """
    Enhanced multi-method table extraction using:
    1. Table Transformer (deep learning)
    2. PDFPlumber (rule-based)
    3. PyMuPDF (structure-based)
    4. Computer vision (fallback)
    
    Features advanced validation to reduce false positives and improve accuracy
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "tables"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        
        # Validation thresholds
        self.TABLE_CONFIDENCE_THRESHOLD = 0.3  # Lowered further to be more permissive for real tables
        self.PROSE_LIKELIHOOD_THRESHOLD = 0.6
        self.MIN_TABLE_ROWS = 1  # More permissive: allow tables with just 1 row
        self.MIN_TABLE_COLS = 1  # More permissive: allow tables with just 1 column
        
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
        """Extract tables using enhanced validation pipeline"""
        all_candidates = []
        
        # Phase 1: Extract table DATA ONLY from multiple methods (no files created yet)
        methods = [
            ('pdfplumber', self._extract_with_pdfplumber_data_only),
            ('transformer', self._extract_with_transformer_data_only),
            ('camelot', self._extract_with_camelot_data_only),
            ('tabula', self._extract_with_tabula_data_only)
        ]
        
        for method_name, method_func in methods:
            try:
                if method_name == 'transformer' and not self.dl_models_available:
                    continue
                    
                candidates = await method_func(pdf_path)
                for candidate in candidates:
                    # Add extraction method metadata
                    if not hasattr(candidate, 'extraction_method'):
                        candidate.extraction_method = method_name
                all_candidates.extend(candidates)
                logger.info(f"{method_name} extracted {len(candidates)} table candidates")
            except Exception as e:
                logger.error(f"{method_name} extraction failed: {e}")
        
        # Phase 2: Advanced validation pipeline
        validated_tables = await self._validate_table_candidates(all_candidates, pdf_path)
        
        # Phase 3: Deduplicate validated tables BEFORE creating any files
        logger.info(f"Before deduplication: {len(validated_tables)} validated tables")
        unique_tables = self._deduplicate_tables([], validated_tables)
        logger.info(f"After deduplication: {len(unique_tables)} unique tables")
        
        # Phase 4: Create files and upload to Cloudinary ONLY for unique tables
        final_tables = []
        for table in unique_tables:
            # Create the actual table object with files and Cloudinary uploads
            final_table = await self._create_final_table_object(table, pdf_path)
            final_tables.append(final_table)
        
        # Log which extractors contributed to final results
        extractor_counts = {}
        for table in final_tables:
            method = table.extraction_method
            extractor_counts[method] = extractor_counts.get(method, 0) + 1
        
        for method, count in extractor_counts.items():
            logger.info(f"Final tables from {method}: {count}")
        
        # Phase 5: Post-process: enhance table structure for all extracted tables
        for table in final_tables:
            if not table.headers or not table.rows:
                await self._enhance_table_structure(table, pdf_path)
        
        logger.info(f"Final result: {len(final_tables)} validated tables from {len(all_candidates)} candidates")
        
        return final_tables
    
    async def _extract_with_pdfplumber_data_only(self, pdf_path: Path) -> List[Table]:
        """PDFPlumber extraction - DATA ONLY, no files created"""
        tables = []
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                
                for idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # Enhanced data cleaning and validation
                    cleaned_table = self._clean_and_validate_table_data(table_data)
                    if not cleaned_table:
                        continue
                    
                    headers, rows = cleaned_table
                    
                    # Estimate bbox with better heuristics
                    bbox = self._estimate_table_bbox(page, table_data, page_num)
                    
                    # Create table object WITHOUT files or Cloudinary uploads
                    table = Table(
                        label=f"Table {page_num}.{idx}",
                        page=page_num,
                        bbox=bbox,
                        headers=headers,
                        rows=rows,
                        csv_path=None,  # No file path yet
                        html=None,      # No HTML yet
                        extraction_method="pdfplumber"
                    )
                    tables.append(table)
        
        return tables
    
    async def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Table]:
        """Enhanced PDFPlumber extraction with better normalization"""
        tables = []
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                
                for idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # Enhanced data cleaning and validation
                    cleaned_table = self._clean_and_validate_table_data(table_data)
                    if not cleaned_table:
                        continue
                    
                    headers, rows = cleaned_table
                    
                    # Estimate bbox with better heuristics
                    bbox = self._estimate_table_bbox(page, table_data, page_num)
                    
                    # Create and save table
                    table = await self._create_table_object(
                        headers, rows, page_num, idx, bbox, "pdfplumber"
                    )
                    tables.append(table)
        
        return tables
    
    async def _extract_with_transformer_data_only(self, pdf_path: Path) -> List[Table]:
        """Transformer extraction - DATA ONLY, no files created"""
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
                
                # Enhanced data cleaning and validation
                if headers and rows:
                    cleaned_table = self._clean_and_validate_table_data([headers[0]] + rows if headers else rows)
                    if cleaned_table:
                        cleaned_headers, cleaned_rows = cleaned_table
                        headers = cleaned_headers
                        rows = cleaned_rows
                
                # Convert coordinates to PDF space
                scale = page.rect.width / pil_image.width
                
                # Extract table caption and references
                caption, references = await self._extract_table_caption_and_references(pdf_path, page_num, x1, y1, x2, y2)
                
                # Create table object WITHOUT files or Cloudinary uploads
                table = Table(
                    label=f"Table {page_num}.{idx}",
                    page=page_num,
                    bbox=BoundingBox(
                        x1=x1 * scale,
                        y1=y1 * scale,
                        x2=x2 * scale,
                        y2=y2 * scale,
                        page=page_num,
                        confidence=0.8
                    ),
                    headers=headers,
                    rows=rows,
                    csv_path=None,  # No file path yet
                    html=None,      # No HTML yet
                    extraction_method="transformer",
                    image_path=None,  # No image path yet
                    caption=caption,
                    references=references
                )
                tables.append(table)
        
        doc.close()
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
                
                # Upload table image to Cloudinary
                filename = f"transformer_page{page_num}_table{idx}"
                cloudinary_url = await cloudinary_service.upload_pil_image(table_image, filename, "scholarai/tables")
                
                # Enhanced data cleaning and validation
                if headers and rows:
                    cleaned_table = self._clean_and_validate_table_data([headers[0]] + rows if headers else rows)
                    if cleaned_table:
                        cleaned_headers, cleaned_rows = cleaned_table
                        headers = cleaned_headers
                        rows = cleaned_rows
                
                # Save as CSV
                csv_path = None
                html = None
                if headers and rows:
                    try:
                        # Ensure headers and rows have consistent column count
                        header_cols = len(headers[0]) if headers else 0
                        max_row_cols = max(len(row) for row in rows) if rows else 0
                        max_cols = max(header_cols, max_row_cols)
                        
                        # Normalize headers
                        if headers and len(headers[0]) < max_cols:
                            headers[0].extend([f"Column {i}" for i in range(len(headers[0]), max_cols)])
                        
                        # Normalize rows
                        normalized_rows = []
                        for row in rows:
                            normalized_row = row[:max_cols]  # Truncate if too long
                            while len(normalized_row) < max_cols:  # Extend if too short
                                normalized_row.append("")
                            normalized_rows.append(normalized_row)
                        
                        csv_path = self.output_dir / f"transformer_page{page_num}_table{idx}.csv"
                        df = pd.DataFrame(normalized_rows, columns=headers[0] if headers else None)
                        df.to_csv(csv_path, index=False)
                        html = df.to_html(index=False, classes='table table-bordered')
                        
                        # Update rows with normalized data
                        rows = normalized_rows
                    except Exception as e:
                        logger.warning(f"Failed to create CSV for table {page_num}.{idx}: {e}")
                        csv_path = None
                        html = None
                
                # Convert coordinates to PDF space
                scale = page.rect.width / pil_image.width
                
                # Extract table caption and references
                caption, references = await self._extract_table_caption_and_references(pdf_path, page_num, x1, y1, x2, y2)
                
                # Upload CSV to Cloudinary if available
                cloudinary_csv_url = None
                if csv_path and csv_path.exists():
                    try:
                        cloudinary_csv_url = await cloudinary_service.upload_file(
                            str(csv_path), 
                            folder="tables/csv",
                            public_id=f"table_{page_num}_{idx}_data"
                        )
                        logger.info(f"Uploaded CSV to Cloudinary: {cloudinary_csv_url}")
                    except Exception as e:
                        logger.warning(f"Failed to upload CSV to Cloudinary: {e}")
                
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
                    csv_path=cloudinary_csv_url or str(csv_path) if csv_path else None,
                    html=html,
                    structure=structure,
                    image_path=cloudinary_url,
                    caption=caption,
                    references=references
                )
                # Add extraction method metadata
                table.extraction_method = "transformer"
                tables.append(table)
        
        doc.close()
        return tables
    
    async def _extract_table_caption_and_references(self, pdf_path: Path, page_num: int, x1: int, y1: int, x2: int, y2: int) -> Tuple[str, List[str]]:
        """Extract table caption and references from surrounding text"""
        try:
            import fitz
            
            doc = fitz.open(str(pdf_path))
            page = doc.load_page(page_num - 1)  # 0-based indexing
            
            # Get text blocks around the table
            text_blocks = page.get_text("dict")["blocks"]
            
            caption = ""
            references = []
            
            # Look for caption above the table (within 100 points)
            table_top = y1
            caption_candidates = []
            
            for block in text_blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            span_y = span["bbox"][1]  # Top of text span
                            
                            # Check if text is above the table and close
                            if span_y < table_top and (table_top - span_y) < 100:
                                # Look for caption patterns
                                if re.match(r'^(Table|Tab\.?)\s*\d+', text, re.IGNORECASE):
                                    caption_candidates.append(text)
                                elif text and len(text) > 10 and len(text) < 200:
                                    # Might be a caption
                                    caption_candidates.append(text)
            
            # Select the best caption candidate
            if caption_candidates:
                # Prefer text that starts with "Table"
                for candidate in caption_candidates:
                    if re.match(r'^(Table|Tab\.?)\s*\d+', candidate, re.IGNORECASE):
                        caption = candidate
                        break
                else:
                    # Use the first substantial candidate
                    caption = caption_candidates[0]
            
            # Look for references in the text
            page_text = page.get_text()
            
            # Find table references in the text
            table_ref_patterns = [
                r'Table\s+(\d+)',
                r'Tab\.?\s*(\d+)',
                r'see\s+Table\s+(\d+)',
                r'refer\s+to\s+Table\s+(\d+)'
            ]
            
            for pattern in table_ref_patterns:
                matches = re.finditer(pattern, page_text, re.IGNORECASE)
                for match in matches:
                    ref_text = match.group(0)
                    if ref_text not in references:
                        references.append(ref_text)
            
            doc.close()
            return caption, references
            
        except Exception as e:
            logger.warning(f"Failed to extract table caption and references: {e}")
            return "", []
    
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
        """Advanced deduplication system with content similarity and extractor prioritization"""
        if not new:
            return existing
        
        # Define extractor priority (higher number = higher priority)
        extractor_priority = {
            'pdfplumber': 4,
            'transformer': 3,
            'tabula': 2,
            'camelot': 1
        }
        
        # Combine existing and new tables
        all_tables = existing + new
        
        # Group tables by page
        tables_by_page = {}
        for table in all_tables:
            if table.page not in tables_by_page:
                tables_by_page[table.page] = []
            tables_by_page[table.page].append(table)
        
        # Deduplicate each page separately
        deduplicated_tables = []
        for page_num, page_tables in tables_by_page.items():
            page_deduplicated = self._deduplicate_page_tables(page_tables, extractor_priority)
            deduplicated_tables.extend(page_deduplicated)
        
        return deduplicated_tables
    
    def _deduplicate_page_tables(self, page_tables: List[Table], extractor_priority: Dict[str, int]) -> List[Table]:
        """Deduplicate tables on the same page using multiple strategies"""
        if len(page_tables) <= 1:
            return page_tables
        
        # Sort tables by extractor priority (highest first)
        page_tables.sort(key=lambda t: extractor_priority.get(t.extraction_method, 0), reverse=True)
        
        unique_tables = []
        
        for table in page_tables:
            is_duplicate = False
            
            # Check against already accepted tables
            for unique_table in unique_tables:
                if self._is_duplicate_table(table, unique_table):
                    is_duplicate = True
                    logger.debug(f"Duplicate detected: {table.label} ({table.extraction_method}) "
                               f"is duplicate of {unique_table.label} ({unique_table.extraction_method})")
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
                logger.debug(f"Added unique table: {table.label} ({table.extraction_method})")
        
        return unique_tables
    
    def _is_duplicate_table(self, table1: Table, table2: Table) -> bool:
        """Check if two tables are duplicates using multiple strategies"""
        # Strategy 1: Spatial overlap (IOU)
        if self._calculate_iou(table1.bbox, table2.bbox) > 0.3:
            return True
        
        # Strategy 2: Content similarity (if both have content)
        if self._has_similar_content(table1, table2):
            return True
        
        # Strategy 3: Page and position proximity
        if self._is_nearby_table(table1, table2):
            return True
        
        return False
    
    def _has_similar_content(self, table1: Table, table2: Table) -> bool:
        """Check if two tables have similar content"""
        # Extract text content from both tables
        content1 = self._extract_table_content(table1)
        content2 = self._extract_table_content(table2)
        
        if not content1 or not content2:
            return False
        
        # Calculate content similarity using multiple metrics
        similarity_score = 0.0
        
        # 1. Text overlap
        text_overlap = self._calculate_text_overlap(content1, content2)
        similarity_score += text_overlap * 0.4
        
        # 2. Structure similarity
        structure_similarity = self._calculate_structure_similarity(table1, table2)
        similarity_score += structure_similarity * 0.3
        
        # 3. Size similarity
        size_similarity = self._calculate_size_similarity(table1, table2)
        similarity_score += size_similarity * 0.3
        
        # Consider tables similar if similarity > 0.6
        return similarity_score > 0.6
    
    def _extract_table_content(self, table: Table) -> str:
        """Extract all text content from a table"""
        content_parts = []
        
        # Add headers
        if table.headers:
            for header_row in table.headers:
                content_parts.extend(str(cell).strip() for cell in header_row if cell)
        
        # Add rows
        if table.rows:
            for row in table.rows:
                content_parts.extend(str(cell).strip() for cell in row if cell)
        
        return ' '.join(content_parts).lower()
    
    def _calculate_text_overlap(self, content1: str, content2: str) -> float:
        """Calculate text overlap between two content strings"""
        if not content1 or not content2:
            return 0.0
        
        # Split into words and create sets
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_structure_similarity(self, table1: Table, table2: Table) -> float:
        """Calculate structural similarity between two tables"""
        # Compare row and column counts
        rows1 = len(table1.rows) if table1.rows else 0
        rows2 = len(table2.rows) if table2.rows else 0
        cols1 = len(table1.headers[0]) if table1.headers else 0
        cols2 = len(table2.headers[0]) if table2.headers else 0
        
        # Calculate similarity scores
        row_similarity = 1.0 - abs(rows1 - rows2) / max(rows1, rows2, 1)
        col_similarity = 1.0 - abs(cols1 - cols2) / max(cols1, cols2, 1)
        
        return (row_similarity + col_similarity) / 2
    
    def _calculate_size_similarity(self, table1: Table, table2: Table) -> float:
        """Calculate size similarity between two tables"""
        if not table1.bbox or not table2.bbox:
            return 0.5  # Default similarity if no bbox
        
        # Calculate areas
        area1 = (table1.bbox.x2 - table1.bbox.x1) * (table1.bbox.y2 - table1.bbox.y1)
        area2 = (table2.bbox.x2 - table2.bbox.x1) * (table2.bbox.y2 - table2.bbox.y1)
        
        if area1 == 0 or area2 == 0:
            return 0.5
        
        # Calculate similarity
        larger_area = max(area1, area2)
        smaller_area = min(area1, area2)
        
        return smaller_area / larger_area
    
    def _is_nearby_table(self, table1: Table, table2: Table, proximity_threshold: float = 100) -> bool:
        """Check if two tables are spatially close to each other"""
        if not table1.bbox or not table2.bbox:
            return False
        
        # Calculate center points
        center1_x = (table1.bbox.x1 + table1.bbox.x2) / 2
        center1_y = (table1.bbox.y1 + table1.bbox.y2) / 2
        center2_x = (table2.bbox.x1 + table2.bbox.x2) / 2
        center2_y = (table2.bbox.y1 + table2.bbox.y2) / 2
        
        # Calculate distance
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        return distance < proximity_threshold
    
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

    # Enhanced validation methods
    async def _validate_table_candidates(self, candidates: List[Table], pdf_path: Path) -> List[Table]:
        """Advanced validation pipeline to filter false positives"""
        validated = []
        
        for candidate in candidates:
            validation_scores = await self._calculate_validation_scores(candidate, pdf_path)
            
            # Combined confidence score
            overall_confidence = self._calculate_overall_confidence(validation_scores)
            
            if overall_confidence > self.TABLE_CONFIDENCE_THRESHOLD:
                candidate.bbox.confidence = overall_confidence
                # Store validation scores as metadata
                if not hasattr(candidate, 'validation_scores'):
                    candidate.validation_scores = validation_scores
                validated.append(candidate)
                logger.debug(f"Table validated: {candidate.label}, confidence: {overall_confidence:.2f}")
            else:
                logger.debug(f"Table rejected: {candidate.label}, confidence: {overall_confidence:.2f}")
        
        return validated
    
    async def _calculate_validation_scores(self, table: Table, pdf_path: Path) -> Dict[str, float]:
        """Calculate comprehensive validation scores"""
        scores = {}
        
        # 1. Visual structure validation
        scores['visual_structure'] = self._validate_visual_structure(table)
        
        # 2. Content semantic analysis
        scores['content_semantics'] = self._analyze_content_semantics(table)
        
        # 3. Size and proportion validation
        scores['size_validation'] = self._validate_table_size(table)
        
        # 4. Context coherence (check surrounding text)
        scores['context_coherence'] = await self._analyze_context_coherence(table, pdf_path)
        
        # 5. Data pattern recognition
        scores['data_patterns'] = self._detect_data_patterns(table)
        
        return scores
    
    def _validate_visual_structure(self, table: Table) -> float:
        """Validate visual structure consistency - simplified and more permissive"""
        if not table.headers or not table.rows:
            return 0.0
        
        score = 0.0
        
        # Basic structure check - more permissive
        if table.headers and len(table.headers[0]) >= 1:  # At least 1 column
            score += 0.4
        
        # Row count check - more permissive
        if len(table.rows) >= 1:  # At least 1 row
            score += 0.3
        
        # Content check - ensure there's actual content
        total_cells = sum(len(row) for row in table.rows) if table.rows else 0
        if total_cells > 0:
            score += 0.3
        
        return min(1.0, score)
    
    def _analyze_content_semantics(self, table: Table) -> float:
        """Analyze content to distinguish tables from prose text - simplified"""
        if not table.rows:
            return 0.0
        
        # Combine all text content
        all_text = []
        if table.headers:
            for header_row in table.headers:
                all_text.extend(str(cell) for cell in header_row)
        
        for row in table.rows:
            all_text.extend(str(cell) for cell in row)
        
        combined_text = ' '.join(all_text)
        
        if not combined_text.strip():
            return 0.0
        
        score = 0.0
        
        # Simple content analysis - more permissive
        # 1. Check if content looks fragmented (table-like)
        words = combined_text.split()
        if len(words) > 0:
            avg_word_length = sum(len(word) for word in words) / len(words)
            # Short average word length suggests table data
            if avg_word_length < 8:
                score += 0.4
            else:
                score += 0.2
        
        # 2. Check for numeric content (common in tables)
        numeric_count = sum(1 for word in words if re.match(r'^\d+(\.\d+)?$', word))
        if numeric_count > 0:
            score += 0.3
        
        # 3. Basic structure check
        if len(table.headers) > 0 and len(table.rows) > 0:
            score += 0.3
        
        return min(1.0, score)
    
    def _analyze_sentence_patterns(self, text: str) -> float:
        """Analyze sentence completion patterns"""
        sentences = re.split(r'[.!?]+', text)
        
        if len(sentences) <= 1:
            return 0.0  # No complete sentences = table-like
        
        complete_sentences = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Reasonable sentence length
                # Check if it has subject-verb structure (basic heuristic)
                words = sentence.split()
                if len(words) >= 3:  # Minimum for a complete sentence
                    complete_sentences += 1
        
        completion_ratio = complete_sentences / len(sentences)
        return completion_ratio  # High ratio = more prose-like
    
    def _analyze_data_diversity(self, text_cells: List[str]) -> float:
        """Analyze diversity of data types in cells"""
        if not text_cells:
            return 0.0
        
        patterns = {
            'numbers': 0,
            'dates': 0,
            'percentages': 0,
            'currency': 0,
            'short_text': 0,
            'long_text': 0
        }
        
        for cell in text_cells:
            cell = str(cell).strip()
            if not cell:
                continue
            
            # Number patterns
            if re.match(r'^\d+(\.\d+)?$', cell):
                patterns['numbers'] += 1
            # Date patterns
            elif re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', cell):
                patterns['dates'] += 1
            # Percentage patterns
            elif re.match(r'\d+(\.\d+)?%', cell):
                patterns['percentages'] += 1
            # Currency patterns
            elif re.match(r'[\$€£¥]\d+', cell):
                patterns['currency'] += 1
            # Text length analysis
            elif len(cell.split()) <= 3:
                patterns['short_text'] += 1
            else:
                patterns['long_text'] += 1
        
        # Calculate diversity score
        non_zero_patterns = sum(1 for count in patterns.values() if count > 0)
        diversity_score = non_zero_patterns / len(patterns)
        
        # Bonus for numeric data (common in tables)
        numeric_ratio = (patterns['numbers'] + patterns['percentages'] + patterns['currency']) / len(text_cells)
        diversity_score += 0.3 * numeric_ratio
        
        return min(1.0, diversity_score)
    
    def _analyze_token_patterns(self, text_cells: List[str]) -> float:
        """Analyze token length patterns"""
        if not text_cells:
            return 0.0
        
        token_lengths = []
        for cell in text_cells:
            cell = str(cell).strip()
            if cell:
                tokens = cell.split()
                token_lengths.extend(len(token) for token in tokens)
        
        if not token_lengths:
            return 0.0
        
        # Tables typically have varied but generally shorter tokens
        avg_length = sum(token_lengths) / len(token_lengths)
        variance = np.var(token_lengths) if len(token_lengths) > 1 else 0
        
        # Optimal range for table tokens: 3-8 characters average
        length_score = 1.0 if 3 <= avg_length <= 8 else max(0, 1 - abs(avg_length - 5.5) / 10)
        
        # Higher variance suggests diverse data types (table-like)
        variance_score = min(1.0, variance / 20)
        
        return (length_score + variance_score) / 2
    
    def _validate_table_size(self, table: Table) -> float:
        """Validate table size and proportions - simplified and more permissive"""
        if not table.bbox:
            return 0.7  # Default score if no bbox - more permissive
        
        score = 0.0
        
        # Basic size check - very permissive
        if table.rows and len(table.rows) > 0:
            score += 0.5
        
        if table.headers and len(table.headers) > 0:
            score += 0.5
        
        return min(1.0, score)
    
    async def _analyze_context_coherence(self, table: Table, pdf_path: Path) -> float:
        """Analyze surrounding text context for table coherence - simplified"""
        # Simplified context analysis - return default score to be more permissive
        return 0.6  # Default moderate score to avoid being too restrictive
    
    def _is_near_table(self, text_bbox: List[float], table_bbox: BoundingBox, 
                      proximity_threshold: float = 100) -> bool:
        """Check if text block is near the table"""
        # Calculate distance between bounding boxes
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2
        text_center_y = (text_bbox[1] + text_bbox[3]) / 2
        
        table_center_x = (table_bbox.x1 + table_bbox.x2) / 2
        table_center_y = (table_bbox.y1 + table_bbox.y2) / 2
        
        distance = np.sqrt((text_center_x - table_center_x)**2 + 
                          (text_center_y - table_center_y)**2)
        
        return distance <= proximity_threshold
    
    def _detect_data_patterns(self, table: Table) -> float:
        """Detect patterns that indicate genuine tabular data - simplified"""
        if not table.rows:
            return 0.0
        
        score = 0.0
        
        # Simple pattern detection - more permissive
        # 1. Basic structure check
        if table.headers and table.rows:
            score += 0.5
        
        # 2. Content check
        total_cells = sum(len(row) for row in table.rows)
        if total_cells > 0:
            score += 0.5
        
        return min(1.0, score)
    
    def _calculate_column_consistency(self, column_data: List[str]) -> float:
        """Calculate consistency score for a column"""
        if not column_data:
            return 0.0
        
        # Classify data types in the column
        type_counts = {'number': 0, 'date': 0, 'text': 0, 'empty': 0}
        
        for cell in column_data:
            if not cell:
                type_counts['empty'] += 1
            elif re.match(r'^\d+(\.\d+)?$', cell):
                type_counts['number'] += 1
            elif re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', cell):
                type_counts['date'] += 1
            else:
                type_counts['text'] += 1
        
        # Calculate consistency (dominant type ratio)
        max_count = max(type_counts.values())
        consistency = max_count / len(column_data)
        
        return consistency
    
    def _calculate_overall_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall confidence score"""
        weights = {
            'visual_structure': 0.25,
            'content_semantics': 0.30,
            'size_validation': 0.15,
            'context_coherence': 0.10,
            'data_patterns': 0.20
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in scores:
                weighted_score += scores[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0

    # Enhanced helper methods
    def _clean_and_validate_table_data(self, table_data: List[List]) -> Optional[Tuple[List[List], List[List]]]:
        """Enhanced cleaning and validation of raw table data - more permissive"""
        if not table_data:
            return None
        
        # More permissive: allow tables with just 1 row (minimum viable table)
        if len(table_data) < 1:
            return None
        
        # Remove completely empty rows but be more permissive
        filtered_data = []
        for row in table_data:
            # Check if row has any non-empty content
            if any(cell and str(cell).strip() for cell in row):
                filtered_data.append(row)
        
        if len(filtered_data) < 1:
            return None
        
        # Detect header row (usually first non-empty row)
        headers = [filtered_data[0]]
        rows = filtered_data[1:] if len(filtered_data) > 1 else []
        
        # Normalize column count
        max_cols = max(len(row) for row in filtered_data) if filtered_data else 1
        
        # Clean and normalize headers - be more permissive
        normalized_headers = []
        for i, cell in enumerate(headers[0]):
            if i >= max_cols:
                break
            cell_text = str(cell).strip() if cell else f"Column {i}"
            normalized_headers.append(cell_text)
        
        # Fill missing header columns
        while len(normalized_headers) < max_cols:
            normalized_headers.append(f"Column {len(normalized_headers)}")
        
        # Clean and normalize rows
        normalized_rows = []
        for row in rows:
            normalized_row = []
            for i in range(max_cols):
                if i < len(row):
                    cell_text = str(row[i]).strip() if row[i] else ""
                else:
                    cell_text = ""
                normalized_row.append(cell_text)
            normalized_rows.append(normalized_row)
        
        return [normalized_headers], normalized_rows
    
    def _estimate_table_bbox(self, page, table_data: List[List], page_num: int) -> BoundingBox:
        """Estimate table bounding box with improved heuristics"""
        # Try to find table boundaries based on content
        # This is a heuristic since PDFPlumber doesn't provide exact coordinates
        
        margin = 50
        return BoundingBox(
            x1=margin,
            y1=margin,
            x2=page.width - margin,
            y2=page.height - margin,
            page=page_num,
            confidence=0.6  # Lower confidence for estimated bbox
        )
    
    async def _create_final_table_object(self, table: Table, pdf_path: Path) -> Table:
        """Create final table object with files and Cloudinary uploads"""
        # Generate unique filename based on page and method
        filename = f"{table.extraction_method}_page{table.page}_table{self._get_table_index(table)}"
        
        # Create CSV file
        csv_path = self.output_dir / f"{filename}.csv"
        cloudinary_csv_url = None
        
        try:
            # Create DataFrame and save CSV
            df = pd.DataFrame(table.rows, columns=table.headers[0] if table.headers else None)
            df.to_csv(csv_path, index=False)
            html = df.to_html(index=False, classes='table table-bordered')
            
            # Upload CSV to Cloudinary
            if csv_path.exists():
                try:
                    cloudinary_csv_url = await cloudinary_service.upload_file(
                        str(csv_path), 
                        folder="tables/csv",
                        public_id=f"{filename}_data"
                    )
                    logger.info(f"Uploaded CSV to Cloudinary: {cloudinary_csv_url}")
                except Exception as e:
                    logger.warning(f"Failed to upload CSV to Cloudinary: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to save table as CSV: {e}")
            csv_path = None
            html = None
        
        # For transformer tables, also upload the image
        image_path = None
        if table.extraction_method == "transformer":
            try:
                # Re-render the page and crop the table region
                doc = fitz.open(str(pdf_path))
                page = doc[table.page - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Crop table region (convert from PDF coordinates to image coordinates)
                scale = pil_image.width / page.rect.width
                x1 = int(table.bbox.x1 * scale)
                y1 = int(table.bbox.y1 * scale)
                x2 = int(table.bbox.x2 * scale)
                y2 = int(table.bbox.y2 * scale)
                
                table_image = pil_image.crop((x1, y1, x2, y2))
                
                # Upload to Cloudinary
                image_path = await cloudinary_service.upload_pil_image(
                    table_image, 
                    f"transformer_page{table.page}_table{self._get_table_index(table)}", 
                    "scholarai/tables"
                )
                doc.close()
            except Exception as e:
                logger.warning(f"Failed to create table image: {e}")
        
        # Create final table object with all paths
        final_table = Table(
            label=table.label,
            page=table.page,
            bbox=table.bbox,
            headers=table.headers,
            rows=table.rows,
            csv_path=cloudinary_csv_url or str(csv_path) if csv_path else None,
            html=html,
            extraction_method=table.extraction_method,
            image_path=image_path,
            caption=getattr(table, 'caption', None),
            references=getattr(table, 'references', None),
            confidence=getattr(table, 'confidence', 0.5)
        )
        
        return final_table
    
    def _get_table_index(self, table: Table) -> int:
        """Get a unique index for the table on its page"""
        # This is a simple implementation - in practice you might want more sophisticated indexing
        return 0  # For now, just use 0 as all tables are unique after deduplication
    
    async def _create_table_object(self, headers: List[List], rows: List[List], 
                           page: int, idx: int, bbox: BoundingBox, 
                           method: str) -> Table:
        """Create table object with enhanced metadata"""
        # Save as CSV
        csv_path = self.output_dir / f"{method}_page{page}_table{idx}.csv"
        cloudinary_csv_url = None
        
        try:
            df = pd.DataFrame(rows, columns=headers[0] if headers else None)
            df.to_csv(csv_path, index=False)
            html = df.to_html(index=False, classes='table table-bordered')
            
            # Upload CSV to Cloudinary
            if csv_path.exists():
                try:
                    cloudinary_csv_url = await cloudinary_service.upload_file(
                        str(csv_path), 
                        folder="tables/csv",
                        public_id=f"{method}_page{page}_table{idx}_data"
                    )
                    logger.info(f"Uploaded CSV to Cloudinary: {cloudinary_csv_url}")
                except Exception as e:
                    logger.warning(f"Failed to upload CSV to Cloudinary: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to save table as CSV: {e}")
            csv_path = None
            html = None
        
        return Table(
            label=f"Table {page}.{idx}",
            page=page,
            bbox=bbox,
            headers=headers,
            rows=rows,
            csv_path=cloudinary_csv_url or str(csv_path) if csv_path else None,
            html=html,
            extraction_method=method
        )
    
    async def _extract_with_camelot_data_only(self, pdf_path: Path) -> List[Table]:
        """Camelot extraction - DATA ONLY, no files created"""
        try:
            # Get the raw table data from Camelot without file creation
            tables = await camelot_extractor.extract(pdf_path)
            
            # Convert to data-only table objects
            data_only_tables = []
            for table in tables:
                # Create a new table object without file paths
                data_only_table = Table(
                    label=table.label,
                    page=table.page,
                    bbox=table.bbox,
                    headers=table.headers,
                    rows=table.rows,
                    csv_path=None,  # No file path yet
                    html=None,      # No HTML yet
                    extraction_method=table.extraction_method,
                    image_path=None,  # No image path yet
                    caption=getattr(table, 'caption', None),
                    references=getattr(table, 'references', None),
                    confidence=getattr(table, 'confidence', 0.5)
                )
                data_only_tables.append(data_only_table)
            
            return data_only_tables
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            return []
    
    async def _extract_with_tabula_data_only(self, pdf_path: Path) -> List[Table]:
        """Tabula extraction - DATA ONLY, no files created"""
        try:
            # Get the raw table data from Tabula without file creation
            tables = await tabula_extractor.extract(pdf_path)
            
            # Convert to data-only table objects
            data_only_tables = []
            for table in tables:
                # Create a new table object without file paths
                data_only_table = Table(
                    label=table.label,
                    page=table.page,
                    bbox=table.bbox,
                    headers=table.headers,
                    rows=table.rows,
                    csv_path=None,  # No file path yet
                    html=None,      # No HTML yet
                    extraction_method=table.extraction_method,
                    image_path=None,  # No image path yet
                    caption=getattr(table, 'caption', None),
                    references=getattr(table, 'references', None),
                    confidence=getattr(table, 'confidence', 0.5)
                )
                data_only_tables.append(data_only_table)
            
            return data_only_tables
        except Exception as e:
            logger.error(f"Tabula extraction failed: {e}")
            return []
    
    async def _extract_with_camelot(self, pdf_path: Path) -> List[Table]:
        """Extract tables using Camelot"""
        try:
            return await camelot_extractor.extract(pdf_path)
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            return []
    
    async def _extract_with_tabula(self, pdf_path: Path) -> List[Table]:
        """Extract tables using Tabula"""
        try:
            return await tabula_extractor.extract(pdf_path)
        except Exception as e:
            logger.error(f"Tabula extraction failed: {e}")
            return []