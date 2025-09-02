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
                                    table_data, page_num, table_idx, "pdfplumber"
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
    
    def _create_table_from_data(self, table_data, page_num: int, table_idx: int, method: str) -> Table:
        """Create Table object from extracted data"""
        # Convert table data to DataFrame for easier processing
        df = pd.DataFrame(table_data)
        
        # Clean up the DataFrame
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Extract headers and rows
        headers = df.iloc[0].tolist() if len(df) > 0 else []
        rows = df.iloc[1:].values.tolist() if len(df) > 1 else []
        
        # Create bounding box (approximate)
        bbox = BoundingBox(
            x1=0, y1=0, x2=100, y2=100, page=page_num, confidence=0.8
        )
        
        return Table(
            id=f"{method}_page{page_num}_table{table_idx}",
            label=f"Table {page_num}.{table_idx}",
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
        """Process table and store it locally and in Cloudinary"""
        try:
            # Generate unique filename
            filename = f"{table.extraction_method}_page{table.page}_table{table.id.split('_')[-1]}"
            
            # Create CSV file
            csv_path = self.output_dir / f"{filename}.csv"
            html_content = None
            
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
            cloudinary_csv_url = None
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
            table.html = html_content
            
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
            
            return table
            
        except Exception as e:
            logger.error(f"Failed to process and store table {table.id}: {e}")
            return None