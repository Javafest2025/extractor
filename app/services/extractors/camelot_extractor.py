#!/usr/bin/env python3
"""
Camelot Table Extractor Service

This service uses Camelot-py to extract tables from PDF documents.
Camelot is particularly good at extracting tables with complex layouts.
"""

import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from app.models.schemas import Table, BoundingBox
from app.services.cloudinary_service import cloudinary_service


class CamelotExtractor:
    """Camelot-based table extractor"""
    
    def __init__(self):
        self.name = "camelot"
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure Camelot is available"""
        if not self._initialized:
            try:
                import camelot
                self._initialized = True
                logger.info("Camelot extractor initialized successfully")
            except ImportError as e:
                logger.error(f"Camelot not available: {e}")
                raise ImportError("Camelot not installed. Install with: pip install camelot-py[cv]")
    
    async def extract(self, pdf_path: Path) -> List[Table]:
        """Extract tables using Camelot"""
        await self._ensure_initialized()
        
        try:
            import camelot
            
            logger.info(f"Extracting tables with Camelot from {pdf_path}")
            
            # Extract tables using Camelot
            tables = camelot.read_pdf(
                str(pdf_path),
                pages='all',
                flavor='lattice',  # Use lattice for tables with borders
                suppress_stdout=True
            )
            
            logger.info(f"Camelot found {len(tables)} table candidates")
            
            extracted_tables = []
            
            for idx, table in enumerate(tables):
                try:
                    # Get table data
                    df = table.df
                    
                    # Skip empty tables
                    if df.empty or df.shape[0] <= 1:
                        continue
                    
                    # Convert DataFrame to list format
                    headers = df.iloc[0].tolist() if not df.empty else []
                    rows = df.iloc[1:].values.tolist() if df.shape[0] > 1 else []
                    
                    # Get table bounding box
                    bbox = table._bbox
                    if bbox:
                        bbox_obj = BoundingBox(
                            x1=bbox[0],
                            y1=bbox[1],
                            x2=bbox[2],
                            y2=bbox[3],
                            page=table.page,
                            confidence=table.accuracy
                        )
                    else:
                        # Fallback bbox
                        bbox_obj = BoundingBox(
                            x1=0, y1=0, x2=100, y2=100,
                            page=table.page,
                            confidence=table.accuracy
                        )
                    
                    # Create table image and upload to Cloudinary
                    image_path = None
                    try:
                        # Get table image
                        table_image = table._image
                        if table_image is not None:
                            # Convert numpy array to PIL Image, then to bytes
                            import io
                            from PIL import Image
                            import numpy as np
                            
                            # Convert numpy array to PIL Image
                            if isinstance(table_image, np.ndarray):
                                pil_image = Image.fromarray(table_image)
                            else:
                                pil_image = table_image
                            
                            img_byte_arr = io.BytesIO()
                            pil_image.save(img_byte_arr, format='PNG')
                            img_byte_arr = img_byte_arr.getvalue()
                            
                            # Upload to Cloudinary
                            image_url = await cloudinary_service.upload_image_from_bytes(
                                img_byte_arr, 
                                f"tables/camelot_page{table.page}_table{idx}",
                                folder="scholarai/tables"
                            )
                            image_path = image_url
                    except Exception as e:
                        logger.warning(f"Failed to process table image: {e}")
                    
                    # Create table object
                    table_obj = Table(
                        label=f"Table {table.page}.{idx}",
                        page=table.page,
                        bbox=bbox_obj,
                        headers=[headers] if headers else [],
                        rows=rows,
                        extraction_method="camelot",
                        image_path=image_path,
                        confidence=table.accuracy / 100.0 if table.accuracy else 0.5
                    )
                    
                    extracted_tables.append(table_obj)
                    logger.debug(f"Camelot extracted table: {table_obj.label}, accuracy: {table.accuracy}%")
                    
                except Exception as e:
                    logger.warning(f"Failed to process Camelot table {idx}: {e}")
                    continue
            
            logger.info(f"Camelot successfully extracted {len(extracted_tables)} tables")
            return extracted_tables
            
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            return []


# Global instance
camelot_extractor = CamelotExtractor()
