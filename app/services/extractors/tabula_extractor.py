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
                        
                        # Create table image (Tabula doesn't provide images directly)
                        # We'll skip image creation for Tabula
                        image_path = None
                        
                        # Create table object
                        table_obj = Table(
                            label=f"Table {page_idx + 1}.{table_idx}",
                            page=page_idx + 1,
                            bbox=bbox_obj,
                            headers=[headers] if headers else [],
                            rows=rows,
                            extraction_method="tabula",
                            image_path=image_path,
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
