# services/pipeline.py
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import traceback
from loguru import logger
import re

from app.models.schemas import (
    ExtractionResult, ExtractionStatus, ExtractionRequest,
    Metadata, Section, Figure, Table, CodeBlock, Equation, Reference, Entity
)
from app.models.enums import EntityType
from app.config import settings
from app.services.extractors.grobid_extractor import GROBIDExtractor
from app.services.extractors.figure_extractor import FigureExtractor
from app.services.extractors.table_extractor import TableExtractor
from app.services.extractors.ocr_math_extractor import OCRMathExtractor
from app.services.extractors.code_extractor import CodeExtractor
from app.utils.exceptions import ExtractionError


class ExtractionPipeline:
    """
    Main pipeline orchestrating all extraction methods
    Implements fallback strategies and error recovery
    """
    
    def __init__(self):
        self.extractors = {}
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize all available extractors"""
        try:
            self.extractors['grobid'] = GROBIDExtractor()
            logger.info("GROBID extractor initialized")
        except Exception as e:
            logger.warning(f"GROBID extractor not available: {e}")
        
        try:
            self.extractors['figure'] = FigureExtractor()
            logger.info("Figure extractor initialized")
        except Exception as e:
            logger.warning(f"Figure extractor initialization failed: {e}")
        
        try:
            self.extractors['table'] = TableExtractor()
            logger.info("Table extractor initialized")
        except Exception as e:
            logger.warning(f"Table extractor initialization failed: {e}")
        
        try:
            self.extractors['ocr_math'] = OCRMathExtractor()
            logger.info("OCR/Math extractor initialized")
        except Exception as e:
            logger.warning(f"OCR/Math extractor initialization failed: {e}")
        
        try:
            self.extractors['code'] = CodeExtractor()
            logger.info("Code extractor initialized")
        except Exception as e:
            logger.warning(f"Code extractor initialization failed: {e}")
    
    async def extract(self, pdf_path: Path, request: ExtractionRequest) -> ExtractionResult:
        """
        Main extraction pipeline
        """
        start_time = datetime.utcnow()
        
        # Initialize result with default metadata
        result = ExtractionResult(
            pdf_path=str(pdf_path),
            pdf_hash=self._calculate_file_hash(pdf_path),
            status=ExtractionStatus.PROCESSING,
            extraction_methods=[],
            metadata=Metadata(title="Unknown", page_count=0)
        )
        
        # Track extraction tasks
        tasks = {}
        errors = []
        warnings = []
        
        try:
            # Phase 1: Text and Structure Extraction
            if request.extract_text and 'grobid' in self.extractors:
                tasks['grobid'] = asyncio.create_task(
                    self._extract_with_grobid(pdf_path)
                )
            
            # Phase 2: Visual Content Extraction (can run in parallel)
            if request.extract_figures and 'figure' in self.extractors:
                tasks['figures'] = asyncio.create_task(
                    self._extract_figures(pdf_path)
                )
            
            if request.extract_tables and 'table' in self.extractors:
                tasks['tables'] = asyncio.create_task(
                    self._extract_tables(pdf_path)
                )
            
            if request.extract_code and 'code' in self.extractors:
                tasks['code'] = asyncio.create_task(
                    self._extract_code(pdf_path)
                )
            
            # Phase 3: OCR and Math (if needed)
            if request.use_ocr and 'ocr_math' in self.extractors:
                tasks['ocr_math'] = asyncio.create_task(
                    self._extract_ocr_math(pdf_path)
                )
            
            # Wait for all tasks with timeout
            timeout = request.timeout or settings.extraction_timeout
            done, pending = await asyncio.wait(
                tasks.values(),
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                warnings.append("Some extraction tasks timed out")
            
            # Collect results
            if 'grobid' in tasks and tasks['grobid'].done():
                try:
                    grobid_result = tasks['grobid'].result()
                    if grobid_result:
                        result.metadata = grobid_result.get('metadata', Metadata(title="Unknown"))
                        result.sections = grobid_result.get('sections', [])
                        result.references = grobid_result.get('references', [])
                        result.extraction_methods.append('grobid')
                except Exception as e:
                    errors.append(f"GROBID extraction failed: {str(e)}")
                    logger.error(f"GROBID error: {traceback.format_exc()}")
            
            if 'figures' in tasks and tasks['figures'].done():
                try:
                    result.figures = tasks['figures'].result()
                    result.extraction_methods.append('pdffigures2_pymupdf_cv')
                except Exception as e:
                    errors.append(f"Figure extraction failed: {str(e)}")
            
            if 'tables' in tasks and tasks['tables'].done():
                try:
                    result.tables = tasks['tables'].result()
                    result.extraction_methods.append('table_transformer_pdfplumber')
                except Exception as e:
                    errors.append(f"Table extraction failed: {str(e)}")
            
            if 'code' in tasks and tasks['code'].done():
                try:
                    result.code_blocks = tasks['code'].result()
                    result.extraction_methods.append('code_detection')
                except Exception as e:
                    errors.append(f"Code extraction failed: {str(e)}")
            
            if 'ocr_math' in tasks and tasks['ocr_math'].done():
                try:
                    ocr_result = tasks['ocr_math'].result()
                    if ocr_result:
                        result.equations = ocr_result.get('equations', [])
                        # Enhance sections with OCR text if needed
                        if ocr_result.get('enhanced_text'):
                            self._enhance_with_ocr(result, ocr_result['enhanced_text'])
                        result.extraction_methods.append('nougat_tesseract')
                except Exception as e:
                    errors.append(f"OCR/Math extraction failed: {str(e)}")
            
            # Phase 4: Post-processing
            
            # Cross-reference figures and tables with text
            self._cross_reference_content(result)
            
            # Extract entities if requested
            if request.detect_entities:
                result.entities = await self._extract_entities(result)
            
            # Calculate quality metrics
            result.extraction_coverage = self._calculate_coverage(result)
            result.confidence_scores = self._calculate_confidence(result)
            
            # Set final status
            if errors:
                result.status = ExtractionStatus.PARTIAL
                result.errors = errors
            else:
                result.status = ExtractionStatus.COMPLETED
            
            result.warnings = warnings
            
        except Exception as e:
            result.status = ExtractionStatus.FAILED
            result.errors = [str(e)]
            logger.error(f"Pipeline failed: {traceback.format_exc()}")
        
        # Calculate processing time
        result.processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Save result to JSON
        await self._save_result(result)
        
        return result
    
    async def _extract_with_grobid(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract using GROBID with fallback"""
        extractor = self.extractors['grobid']
        return await extractor.extract(pdf_path)
    
    async def _extract_figures(self, pdf_path: Path) -> List[Figure]:
        """Extract figures"""
        extractor = self.extractors['figure']
        return await extractor.extract(pdf_path)
    
    async def _extract_tables(self, pdf_path: Path) -> List[Table]:
        """Extract tables"""
        extractor = self.extractors['table']
        return await extractor.extract(pdf_path)
    
    async def _extract_code(self, pdf_path: Path) -> List[CodeBlock]:
        """Extract code blocks"""
        extractor = self.extractors['code']
        return await extractor.extract(pdf_path)
    
    async def _extract_ocr_math(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract using OCR and math detection"""
        extractor = self.extractors['ocr_math']
        return await extractor.extract(pdf_path)
    
    def _enhance_with_ocr(self, result: ExtractionResult, ocr_text: List[Dict]):
        """Enhance extraction result with OCR text"""
        # If no sections were extracted, create from OCR
        if not result.sections and ocr_text:
            for page_data in ocr_text:
                section = Section(
                    title=f"Page {page_data['page']}",
                    page_start=page_data['page'],
                    page_end=page_data['page'],
                    paragraphs=[{
                        'text': page_data['text'],
                        'page': page_data['page']
                    }]
                )
                result.sections.append(section)
    
    def _cross_reference_content(self, result: ExtractionResult):
        """Cross-reference figures, tables, and code with text"""
        # Build reference map
        figure_refs = {}
        table_refs = {}
        
        # Find references in text
        for section in result.sections:
            for para in section.paragraphs:
                text = para.text if isinstance(para, dict) else para.text
                
                # Find figure references
                fig_patterns = [
                    r'Figure\s+(\d+)',
                    r'Fig\.\s*(\d+)',
                    r'figure\s+(\d+)'
                ]
                for pattern in fig_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        fig_num = match.group(1)
                        if fig_num not in figure_refs:
                            figure_refs[fig_num] = []
                        figure_refs[fig_num].append(section.id)
                
                # Find table references
                table_patterns = [
                    r'Table\s+(\d+)',
                    r'table\s+(\d+)'
                ]
                for pattern in table_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        table_num = match.group(1)
                        if table_num not in table_refs:
                            table_refs[table_num] = []
                        table_refs[table_num].append(section.id)
        
        # Update figures with references
        for figure in result.figures:
            if figure.label:
                # Extract number from label
                match = re.search(r'\d+', figure.label)
                if match:
                    fig_num = match.group()
                    if fig_num in figure_refs:
                        figure.references = figure_refs[fig_num]
        
        # Update tables with references
        for table in result.tables:
            if table.label:
                match = re.search(r'\d+', table.label)
                if match:
                    table_num = match.group()
                    if table_num in table_refs:
                        table.references = table_refs[table_num]
    
    async def _extract_entities(self, result: ExtractionResult) -> List[Entity]:
        """Extract named entities from the document"""
        entities = []
        
        # Extract dataset mentions
        dataset_patterns = [
            r'(?:dataset|corpus|benchmark):\s*(\w+)',
            r'(\w+)\s+dataset',
            r'(?:MNIST|CIFAR|ImageNet|COCO|WikiText|GLUE|SQuAD)'
        ]
        
        # Extract code/tool mentions
        code_patterns = [
            r'(?:github\.com/[\w-]+/[\w-]+)',
            r'(?:implementation|code):\s*([\w-]+)',
            r'(?:PyTorch|TensorFlow|JAX|scikit-learn|NumPy|Pandas)'
        ]
        
        # Extract method mentions
        method_patterns = [
            r'(?:algorithm|method|approach):\s*([\w\s]+)',
            r'(?:BERT|GPT|ResNet|LSTM|GRU|CNN|RNN|Transformer)'
        ]
        
        # Search in sections
        for section in result.sections:
            for para in section.paragraphs:
                text = para.text if isinstance(para, dict) else str(para.text)
                page = para.page if hasattr(para, 'page') else section.page_start
                
                # Find datasets
                for pattern in dataset_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        entity = Entity(
                            type=EntityType.DATASET,
                            name=match.group(1) if match.lastindex else match.group(),
                            page=page,
                            context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                        )
                        entities.append(entity)
                
                # Find code repositories
                for pattern in code_patterns:
                    for match in re.finditer(pattern, text):
                        entity = Entity(
                            type=EntityType.CODE,
                            name=match.group(1) if match.lastindex else match.group(),
                            page=page,
                            context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                        )
                        entities.append(entity)
        
        # Deduplicate entities
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity.type, entity.name.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _calculate_coverage(self, result: ExtractionResult) -> float:
        """Calculate extraction coverage percentage"""
        score = 0.0
        max_score = 100.0
        
        # Check metadata (10%)
        if result.metadata and result.metadata.title != "Unknown":
            score += 10
        
        # Check sections (30%)
        if result.sections:
            section_score = min(30, len(result.sections) * 3)
            score += section_score
        
        # Check figures (15%)
        if result.figures:
            figure_score = min(15, len(result.figures) * 3)
            score += figure_score
        
        # Check tables (15%)
        if result.tables:
            table_score = min(15, len(result.tables) * 5)
            score += table_score
        
        # Check code blocks (10%)
        if result.code_blocks:
            code_score = min(10, len(result.code_blocks) * 2)
            score += code_score
        
        # Check equations (10%)
        if result.equations:
            eq_score = min(10, len(result.equations) * 1)
            score += eq_score
        
        # Check references (10%)
        if result.references:
            ref_score = min(10, len(result.references) * 0.5)
            score += ref_score
        
        return min(100.0, score)
    
    def _calculate_confidence(self, result: ExtractionResult) -> Dict[str, float]:
        """Calculate confidence scores for different extraction types"""
        confidence = {}
        
        # Text confidence (based on extraction method)
        if 'grobid' in result.extraction_methods:
            confidence['text'] = 0.9
        elif 'nougat_tesseract' in result.extraction_methods:
            confidence['text'] = 0.7
        else:
            confidence['text'] = 0.5
        
        # Figure confidence
        if result.figures:
            avg_conf = sum(f.bbox.confidence for f in result.figures if f.bbox and f.bbox.confidence) 
            count = sum(1 for f in result.figures if f.bbox and f.bbox.confidence)
            confidence['figures'] = avg_conf / count if count > 0 else 0.8
        
        # Table confidence
        if result.tables:
            avg_conf = sum(t.bbox.confidence for t in result.tables if t.bbox and t.bbox.confidence)
            count = sum(1 for t in result.tables if t.bbox and t.bbox.confidence)
            confidence['tables'] = avg_conf / count if count > 0 else 0.8
        
        # Code confidence
        confidence['code'] = 0.85 if result.code_blocks else 0.0
        
        # Math confidence
        confidence['equations'] = 0.9 if 'nougat' in str(result.extraction_methods) else 0.7
        
        return confidence
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    async def _save_result(self, result: ExtractionResult):
        """Save extraction result to JSON file"""
        output_path = settings.paper_folder / f"{Path(result.pdf_path).stem}_extraction.json"
        
        # Convert to dict
        result_dict = result.dict()
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Extraction result saved to {output_path}")