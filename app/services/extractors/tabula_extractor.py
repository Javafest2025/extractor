#!/usr/bin/env python3
"""
Tabula Table Extractor Service

This service uses Tabula-py to extract tables from PDF documents.
Tabula is good at extracting tables from various PDF formats.
"""

import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from app.models.schemas import Table, BoundingBox
from app.services.cloudinary_service import cloudinary_service


class TabulaExtractor:
    """Tabula-based table extractor"""
    
    def __init__(self):
        self.name = "tabula"
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure Tabula is available"""
        if not self._initialized:
            try:
                import tabula
                self._initialized = True
                logger.info("Tabula extractor initialized successfully")
            except ImportError as e:
                logger.error(f"Tabula not available: {e}")
                raise ImportError("Tabula not installed. Install with: pip install tabula-py")
    
    def _extract_caption_for_tabula_table(self, pdf_path: Path, page_num: int, table_idx: int) -> str:
        """Extract caption for Tabula tables using PDFPlumber for text extraction"""
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    
                    # Search for table caption patterns
                    caption_text = self._search_table_caption_patterns(page, page_num, table_idx)
                    
                    return caption_text.strip()
                    
        except Exception as e:
            logger.debug(f"Caption extraction failed for Tabula table {page_num}.{table_idx}: {e}")
            return ""
        
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
    
    async def extract(self, pdf_path: Path) -> List[Table]:
        """Extract tables using Tabula"""
        await self._ensure_initialized()
        
        try:
            import tabula
            
            logger.info(f"Extracting tables with Tabula from {pdf_path}")
            
            # Extract tables using Tabula
            try:
                tables = tabula.read_pdf(
                    str(pdf_path),
                    pages='all',
                    multiple_tables=True,
                    guess=False,  # Don't guess table structure
                    lattice=True,  # Use lattice for tables with borders
                    stream=True,   # Use stream for tables without borders
                    pandas_options={'header': None},  # Don't assume first row is header
                    encoding='utf-8'  # Specify encoding
                )
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                logger.warning("UTF-8 encoding failed, trying with latin-1")
                tables = tabula.read_pdf(
                    str(pdf_path),
                    pages='all',
                    multiple_tables=True,
                    guess=False,
                    lattice=True,
                    stream=True,
                    pandas_options={'header': None},
                    encoding='latin-1'
                )
            
            logger.info(f"Tabula found {len(tables)} table candidates")
            
            extracted_tables = []
            
            for page_idx, page_tables in enumerate(tables):
                # Handle case where page_tables might not be a list
                if not isinstance(page_tables, list):
                    page_tables = [page_tables]
                
                for table_idx, df in enumerate(page_tables):
                    try:
                        # Skip empty tables or non-DataFrame objects
                        if not hasattr(df, 'empty') or df.empty or df.shape[0] <= 1:
                            continue
                        
                        # Convert DataFrame to list format and handle NaN values
                        import pandas as pd
                        
                        # Handle headers - replace NaN with empty string
                        if not df.empty:
                            headers = df.iloc[0].fillna('').astype(str).tolist()
                        else:
                            headers = []
                        
                        # Handle rows - replace NaN with empty string
                        if df.shape[0] > 1:
                            rows_data = df.iloc[1:].fillna('').astype(str)
                            rows = rows_data.values.tolist()
                        else:
                            rows = []
                        
                        # Create bounding box (Tabula doesn't provide precise bbox)
                        # We'll use a generic bbox for the page
                        bbox_obj = BoundingBox(
                            x1=50, y1=50, x2=550, y2=750,  # Generic page area
                            page=page_idx + 1,
                            confidence=0.7  # Default confidence
                        )
                        
                        # Try to extract caption for Tabula tables
                        caption = self._extract_caption_for_tabula_table(pdf_path, page_idx + 1, table_idx)
                        
                        # Create table object WITHOUT files or Cloudinary uploads
                        table_obj = Table(
                            label=f"Table {page_idx + 1}.{table_idx}",
                            caption=caption,
                            page=page_idx + 1,
                            bbox=bbox_obj,
                            headers=[headers] if headers else [],
                            rows=rows,
                            csv_path=None,  # No file path yet
                            extraction_method="tabula",
                            image_path=None,  # No image path yet
                            confidence=0.7  # Default confidence
                        )
                        
                        extracted_tables.append(table_obj)
                        logger.debug(f"Tabula extracted table: {table_obj.label}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process Tabula table {page_idx}.{table_idx}: {e}")
                        continue
            
            logger.info(f"Tabula successfully extracted {len(extracted_tables)} tables")
            return extracted_tables
            
        except Exception as e:
            logger.error(f"Tabula extraction failed: {e}")
            return []


# Global instance
tabula_extractor = TabulaExtractor()
