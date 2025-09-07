# services/extractors/table_extractor.py
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from loguru import logger
import json
import io
import re

from app.models.schemas import Table, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError
from app.services.cloudinary_service import cloudinary_service

from app.services.extractors.tabula_extractor import tabula_extractor


class TableExtractor:
    """
    Lightweight table extraction using rule-based methods only:
    1. PDFPlumber (primary method)
    2. Tabula (fallback only if PDFPlumber extracts 0 tables)
    
    AI models disabled for memory optimization
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "tables"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Validation thresholds
        self.TABLE_CONFIDENCE_THRESHOLD = 0.3
        self.MIN_TABLE_ROWS = 1
        self.MIN_TABLE_COLS = 1
        
        # AI models disabled for memory optimization
        self.dl_models_available = False
        logger.info("Table extractor initialized (AI models disabled for memory optimization)")
    
    async def extract(self, pdf_path: Path) -> List[Table]:
        """Extract tables using lightweight rule-based methods only"""
        logger.info(f"Extracting tables from {pdf_path} using rule-based methods")
        
        all_tables = []
        
        # Method 1: PDFPlumber (primary method)
        try:
            pdfplumber_tables = await self._extract_with_pdfplumber(pdf_path)
            all_tables.extend(pdfplumber_tables)
            logger.info(f"PDFPlumber found {len(pdfplumber_tables)} tables")
        except Exception as e:
            logger.warning(f"PDFPlumber extraction failed: {e}")
        
        # Method 2: Tabula (fallback only if PDFPlumber extracts 0 tables)
        if len(pdfplumber_tables) == 0:
            try:
                tabula_tables = await tabula_extractor.extract(pdf_path)
                all_tables.extend(tabula_tables)
                logger.info(f"Tabula fallback found {len(tabula_tables)} tables")
            except Exception as e:
                logger.warning(f"Tabula fallback extraction failed: {e}")
        else:
            logger.info("PDFPlumber extracted tables, skipping Tabula fallback")
        
        # Deduplicate and validate tables
        unique_tables = self._deduplicate_tables(all_tables)
        validated_tables = [table for table in unique_tables if self._validate_table(table)]
        
        logger.info(f"Total tables extracted: {len(validated_tables)}")
        
        # Process and store tables
        final_tables = []
        for table in validated_tables:
            try:
                processed_table = await self._process_and_store_table(table, pdf_path)
                if processed_table:
                    final_tables.append(processed_table)
            except Exception as e:
                logger.error(f"Failed to process table {table.id}: {e}")
        
        logger.info(f"Successfully processed and stored {len(final_tables)} tables")
        return final_tables
    
    async def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Table]:
        """Extract tables using PDFPlumber"""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table_idx, table_data in enumerate(page_tables):
                            if self._is_valid_table_data(table_data):
                                table = self._create_table_from_data(
                                    table_data, page_num, table_idx, "pdfplumber", page
                                )
                                tables.append(table)
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
        
        return tables
    

    
    def _is_valid_table_data(self, table_data) -> bool:
        """Check if table data is valid"""
        if not table_data or not isinstance(table_data, (list, tuple)):
            return False
        
        # Check if table has reasonable dimensions
        if len(table_data) < 1:
            return False
        
        # Check if table has content
        has_content = False
        for row in table_data:
            if row and any(cell and str(cell).strip() for cell in row):
                has_content = True
                break
        
        return has_content
    
    def _extract_caption_near_table(self, page, table_data, page_num: int, table_idx: int) -> str:
        """Extract caption text near the table"""
        try:
            caption_text = ""
            
            # Get table position information from PDFPlumber
            if hasattr(page, 'find_tables'):
                # Try to find the specific table to get its position
                tables = page.find_tables()
                if table_idx < len(tables):
                    table_bbox = tables[table_idx].bbox
                    if table_bbox:
                        # Extract caption below the table (most common)
                        caption_text = self._search_caption_below_table(page, table_bbox)
                        
                        # If no caption below, search above
                        if not caption_text.strip():
                            caption_text = self._search_caption_above_table(page, table_bbox)
            
            # Fallback: search for text patterns that look like table captions
            if not caption_text.strip():
                caption_text = self._search_table_caption_patterns(page, page_num, table_idx)
            
            return caption_text.strip()
            
        except Exception as e:
            logger.warning(f"Caption extraction failed for table {page_num}.{table_idx}: {e}")
            return ""
    
    def _search_caption_below_table(self, page, table_bbox) -> str:
        """Search for caption text below the table"""
        try:
            caption_text = ""
            
            # Search for text below the table
            for word in page.extract_words():
                word_x0, word_top, word_x1, word_bottom = word['x0'], word['top'], word['x1'], word['bottom']
                
                # Check if word is below table (within reasonable distance)
                if (word_top > table_bbox[3] and word_top < table_bbox[3] + 150 and
                    abs(word_x0 - (table_bbox[0] + table_bbox[2]) / 2) < 100):
                    caption_text += word['text'] + " "
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Below table caption search failed: {e}")
            return ""
    
    def _search_caption_above_table(self, page, table_bbox) -> str:
        """Search for caption text above the table"""
        try:
            caption_text = ""
            
            # Search for text above the table
            for word in page.extract_words():
                word_x0, word_top, word_x1, word_bottom = word['x0'], word['top'], word['x1'], word['bottom']
                
                # Check if word is above table (within reasonable distance)
                if (word_bottom < table_bbox[1] and word_bottom > table_bbox[1] - 150 and
                    abs(word_x0 - (table_bbox[0] + table_bbox[2]) / 2) < 100):
                    caption_text += word['text'] + " "
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Above table caption search failed: {e}")
            return ""
    
    def _search_table_caption_patterns(self, page, page_num: int, table_idx: int) -> str:
        """Search for table caption patterns in text"""
        try:
            caption_text = ""
            
            # Look for common table caption patterns
            caption_patterns = [
                f"Table {page_num}.{table_idx}",
                f"Table {page_num}.{table_idx + 1}",  # Sometimes 1-indexed
                f"TABLE {page_num}.{table_idx}",
                f"TABLE {page_num}.{table_idx + 1}",
                f"Table {page_num}",
                f"TABLE {page_num}"
            ]
            
            # Extract all text from the page
            page_text = page.extract_text()
            if page_text:
                # Look for caption patterns
                for pattern in caption_patterns:
                    if pattern in page_text:
                        # Extract text around the pattern
                        pattern_index = page_text.find(pattern)
                        start = max(0, pattern_index - 200)
                        end = min(len(page_text), pattern_index + 200)
                        caption_text = page_text[start:end].strip()
                        break
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Pattern-based caption search failed: {e}")
            return ""
    
    def _create_table_from_data(self, table_data, page_num: int, table_idx: int, method: str, page=None) -> Table:
        """Create Table object from extracted data"""
        # Convert table data to DataFrame for easier processing
        df = pd.DataFrame(table_data)
        
        # Clean up the DataFrame
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Extract headers and rows, ensuring no None values
        headers = []
        if len(df) > 0:
            headers = [str(cell) if cell is not None else "" for cell in df.iloc[0].tolist()]
        
        rows = []
        if len(df) > 1:
            rows = [[str(cell) if cell is not None else "" for cell in row] for row in df.iloc[1:].values.tolist()]
        
        # Create bounding box (approximate)
        bbox = BoundingBox(
            x1=0, y1=0, x2=100, y2=100, page=page_num, confidence=0.8
        )
        
        # Extract caption if page object is provided
        caption = None
        if page:
            caption = self._extract_caption_near_table(page, table_data, page_num, table_idx)
        
        return Table(
            id=f"{method}_page{page_num}_table{table_idx}",
            label=f"Table {page_num}.{table_idx}",
            caption=caption,
            page=page_num,
            bbox=bbox,
            headers=[headers] if headers else [],
            rows=rows,
            extraction_method=method,
            csv_path=None,
            html=None,
            image_path=None
        )
    
    def _validate_table(self, table: Table) -> bool:
        """Validate extracted table"""
        if not table.headers or not table.rows:
            return False
        
        # Check minimum dimensions
        if len(table.headers) < self.MIN_TABLE_COLS or len(table.rows) < self.MIN_TABLE_ROWS:
            return False
        
        # Check content density
        total_cells = len(table.headers) * len(table.rows) if table.rows else 0
        if total_cells > 0:
            content_cells = len([cell for row in table.rows for cell in row if cell and str(cell).strip()])
            content_density = content_cells / total_cells
            if content_density < 0.1:  # At least 10% of cells should have content
                return False
        
        return True
    
    def _deduplicate_tables(self, tables: List[Table]) -> List[Table]:
        """Remove duplicate tables based on content similarity"""
        unique_tables = []
        seen_content = set()
        
        for table in tables:
            # Create a content hash
            content_str = f"{table.page}_{len(table.headers)}_{len(table.rows)}"
            
            if content_str not in seen_content:
                seen_content.add(content_str)
                unique_tables.append(table)
        
        return unique_tables
    
    async def _process_and_store_table(self, table: Table, pdf_path: Path) -> Optional[Table]:
        """Process table and store it based on STORE_LOCALLY setting"""
        try:
            # Generate unique filename
            filename = f"{table.extraction_method}_page{table.page}_table{table.id.split('_')[-1]}"
            
            html_content = None
            cloudinary_csv_url = None
            
            if settings.store_locally:
                # Store locally if enabled
                csv_path = self.output_dir / f"{filename}.csv"
                try:
                    # Create DataFrame and save CSV
                    df = pd.DataFrame(table.rows, columns=table.headers[0] if table.headers else None)
                    df.to_csv(csv_path, index=False)
                    html_content = df.to_html(index=False, classes='table table-bordered')
                    
                    logger.info(f"Created CSV file: {csv_path}")
                except Exception as e:
                    logger.error(f"Failed to create CSV for table {table.id}: {e}")
                    return None
                
                # Upload CSV to Cloudinary
                try:
                    if csv_path.exists():
                        cloudinary_csv_url = await cloudinary_service.upload_file(
                            str(csv_path), 
                            folder="tables/csv",
                            public_id=f"{filename}_data"
                        )
                        logger.info(f"Uploaded CSV to Cloudinary: {cloudinary_csv_url}")
                except Exception as e:
                    logger.warning(f"Failed to upload CSV to Cloudinary: {e}")
                    
                # Update table object with file paths
                table.csv_path = cloudinary_csv_url or str(csv_path)
                
                # Save table metadata to JSON
                json_path = self.output_dir / f"{filename}.json"
                try:
                    table_dict = {
                        "id": table.id,
                        "page": table.page,
                        "extraction_method": table.extraction_method,
                        "headers": table.headers,
                        "rows": table.rows,
                        "csv_path": table.csv_path,
                        "extraction_timestamp": pd.Timestamp.now().isoformat()
                    }
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(table_dict, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Saved table metadata: {json_path}")
                except Exception as e:
                    logger.error(f"Failed to save table metadata: {e}")
            else:
                # Only store in Cloudinary, not locally
                try:
                    # Create DataFrame in memory
                    df = pd.DataFrame(table.rows, columns=table.headers[0] if table.headers else None)
                    html_content = df.to_html(index=False, classes='table table-bordered')
                    
                    # Convert DataFrame to CSV bytes for Cloudinary upload
                    csv_buffer = io.BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_bytes = csv_buffer.getvalue()
                    
                    # Upload CSV bytes to Cloudinary
                    cloudinary_csv_url = await cloudinary_service.upload_bytes(
                        csv_bytes, 
                        folder="tables/csv",
                        public_id=f"{filename}_data",
                        file_extension=".csv"
                    )
                    logger.info(f"Uploaded CSV to Cloudinary: {cloudinary_csv_url}")
                    
                    # Update table object with Cloudinary URL
                    table.csv_path = cloudinary_csv_url
                    
                except Exception as e:
                    logger.error(f"Failed to upload CSV to Cloudinary: {e}")
                    return None
            
            # Set HTML content
            table.html = html_content
            
            return table
            
        except Exception as e:
            logger.error(f"Failed to process and store table {table.id}: {e}")
            return None