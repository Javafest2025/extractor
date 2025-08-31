
# services/extractors/grobid_extractor.py
import httpx
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from loguru import logger

from app.models.schemas import (
    Metadata, Section, Reference, Author, 
    Paragraph, SectionType
)
from app.config import settings
from app.utils.exceptions import ExtractionError


class GROBIDExtractor:
    """
    GROBID service integration for extracting structured content from academic PDFs.
    Handles: metadata, sections, paragraphs, references
    """
    
    NAMESPACES = {
        'tei': 'http://www.tei-c.org/ns/1.0'
    }
    
    SECTION_MAPPING = {
        'abstract': SectionType.ABSTRACT,
        'introduction': SectionType.INTRODUCTION,
        'related': SectionType.RELATED_WORK,
        'method': SectionType.METHODOLOGY,
        'experiment': SectionType.EXPERIMENTS,
        'result': SectionType.RESULTS,
        'discussion': SectionType.DISCUSSION,
        'conclusion': SectionType.CONCLUSION,
        'appendix': SectionType.APPENDIX,
    }
    
    def __init__(self, grobid_url: str = None):
        self.grobid_url = grobid_url or settings.grobid_url
        self.client = None  # Initialize client lazily
        self._service_available = None  # Cache service availability
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized"""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=180.0)
    
    async def check_service(self) -> bool:
        """Check if GROBID service is available"""
        if self._service_available is not None:
            return self._service_available
            
        try:
            await self._ensure_client()
            response = await self.client.get(f"{self.grobid_url}/api/isalive")
            self._service_available = response.status_code == 200
            return self._service_available
        except Exception as e:
            logger.warning(f"GROBID service unavailable: {e}")
            self._service_available = False
            return False
    
    async def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract structured content from PDF using GROBID
        
        Returns:
            Dictionary containing metadata, sections, and references
        """
        # Get page count from PDF
        page_count = self._get_pdf_page_count(pdf_path)
        
        if not await self.check_service():
            # Return empty result instead of raising error
            logger.warning("GROBID service not available, returning empty result")
            return {
                'metadata': Metadata(title="Unknown", page_count=page_count),
                'sections': [],
                'references': [],
                'raw_tei': None
            }
        
        try:
            # Store PDF path for page count fallback
            self.pdf_path = pdf_path
            
            # Process full document
            tei_xml = await self._process_fulltext(pdf_path)
            
            # Parse TEI XML
            root = ET.fromstring(tei_xml)
            
            # Extract components
            metadata = self._extract_metadata(root)
            sections = self._extract_sections(root)
            references = self._extract_references(root)
            
            return {
                'metadata': metadata,
                'sections': sections,
                'references': references,
                'raw_tei': tei_xml  # Keep for debugging
            }
            
        except Exception as e:
            logger.error(f"GROBID extraction failed: {e}")
            raise ExtractionError(f"GROBID extraction failed: {str(e)}")
    
    async def _process_fulltext(self, pdf_path: Path) -> str:
        """Process PDF with GROBID fulltext endpoint"""
        await self._ensure_client()
        
        with open(pdf_path, 'rb') as f:
            files = {'input': (pdf_path.name, f, 'application/pdf')}
            
            response = await self.client.post(
                f"{self.grobid_url}/api/processFulltextDocument",
                files=files,
                data={
                    'consolidateHeader': '1',
                    'consolidateCitations': '1',
                    'includeRawCitations': '1',
                    'includeRawAffiliations': '1',
                    'teiCoordinates': 'true'
                }
            )
            
            if response.status_code != 200:
                raise ExtractionError(f"GROBID API error: {response.status_code}")
            
            return response.text
    
    def _extract_metadata(self, root: ET.Element) -> Metadata:
        """Extract metadata from TEI header"""
        header = root.find('.//tei:teiHeader', self.NAMESPACES)
        if header is None:
            return Metadata(title="Unknown")
        
        # Title
        title_elem = header.find('.//tei:titleStmt/tei:title', self.NAMESPACES)
        title = title_elem.text if title_elem is not None else "Unknown"
        
        # Authors
        authors = []
        for author_elem in header.findall('.//tei:author', self.NAMESPACES):
            name_parts = []
            forename = author_elem.find('.//tei:forename', self.NAMESPACES)
            surname = author_elem.find('.//tei:surname', self.NAMESPACES)
            
            if forename is not None:
                name_parts.append(forename.text)
            if surname is not None:
                name_parts.append(surname.text)
            
            if name_parts:
                name = ' '.join(name_parts)
                
                # Affiliation
                affiliation = None
                aff_elem = author_elem.find('.//tei:affiliation', self.NAMESPACES)
                if aff_elem is not None:
                    org_name = aff_elem.find('.//tei:orgName', self.NAMESPACES)
                    if org_name is not None:
                        affiliation = org_name.text
                
                # Email
                email = None
                email_elem = author_elem.find('.//tei:email', self.NAMESPACES)
                if email_elem is not None:
                    email = email_elem.text
                
                authors.append(Author(
                    name=name,
                    affiliation=affiliation,
                    email=email
                ))
        
        # Abstract
        abstract = None
        abstract_elem = root.find('.//tei:abstract', self.NAMESPACES)
        if abstract_elem is not None:
            abstract_parts = []
            for p in abstract_elem.findall('.//tei:p', self.NAMESPACES):
                if p.text:
                    abstract_parts.append(p.text.strip())
            abstract = ' '.join(abstract_parts)
        
        # Keywords
        keywords = []
        for kw in root.findall('.//tei:keywords/tei:term', self.NAMESPACES):
            if kw.text:
                keywords.append(kw.text)
        
        # DOI
        doi = None
        idno = header.find('.//tei:idno[@type="DOI"]', self.NAMESPACES)
        if idno is not None:
            doi = idno.text
        
        # Year
        year = None
        date_elem = header.find('.//tei:date', self.NAMESPACES)
        if date_elem is not None and date_elem.get('when'):
            try:
                year = int(date_elem.get('when')[:4])
            except:
                pass
        
        # Get page count from TEI or fallback to PDF
        page_count = self._get_pdf_page_count_from_tei(root)
        if page_count == 0:
            # Fallback to getting page count from PDF file
            page_count = self._get_pdf_page_count(self.pdf_path) if hasattr(self, 'pdf_path') else 0
        
        return Metadata(
            title=title,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            doi=doi,
            year=year,
            page_count=page_count
        )
    
    def _get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF file"""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            logger.warning(f"Failed to get page count for {pdf_path}: {e}")
            return 0
    
    def _get_pdf_page_count_from_tei(self, root: ET.Element) -> int:
        """Extract page count from TEI XML if available"""
        try:
            # Look for page count in TEI header
            header = root.find('.//tei:teiHeader', self.NAMESPACES)
            if header is not None:
                # Check for page count in various possible locations
                page_count_elem = header.find('.//tei:extent/tei:measure[@unit="page"]', self.NAMESPACES)
                if page_count_elem is not None and page_count_elem.text:
                    try:
                        return int(page_count_elem.text)
                    except ValueError:
                        pass
                
                # Alternative: count pages in body
                body = root.find('.//tei:body', self.NAMESPACES)
                if body is not None:
                    pages = body.findall('.//tei:pb', self.NAMESPACES)
                    if pages:
                        return len(pages) + 1  # +1 because first page doesn't have pb element
        except Exception as e:
            logger.warning(f"Failed to extract page count from TEI: {e}")
        
        return 0
    
    def _extract_sections(self, root: ET.Element) -> List[Section]:
        """Extract document sections from TEI body"""
        sections = []
        body = root.find('.//tei:body', self.NAMESPACES)
        
        if body is None:
            return sections
        
        section_id = 0
        for div in body.findall('.//tei:div', self.NAMESPACES):
            section = self._parse_section(div, section_id)
            if section:
                sections.append(section)
                section_id += 1
        
        return sections
    
    def _parse_section(self, div: ET.Element, section_id: int) -> Optional[Section]:
        """Parse a single section from TEI div element"""
        # Get section title
        head = div.find('tei:head', self.NAMESPACES)
        if head is None:
            return None
        
        title = head.text or f"Section {section_id + 1}"
        
        # Determine section type
        section_type = SectionType.OTHER
        title_lower = title.lower()
        for keyword, stype in self.SECTION_MAPPING.items():
            if keyword in title_lower:
                section_type = stype
                break
        
        # Extract paragraphs with improved page detection
        paragraphs = []
        page_nums = set()
        
        for p_elem in div.findall('tei:p', self.NAMESPACES):
            text = self._extract_text_from_element(p_elem)
            if text:
                # Use improved page detection
                page = self._get_accurate_page_number(p_elem, text)
                page_nums.add(page)
                paragraphs.append(Paragraph(
                    text=text,
                    page=page
                ))
        
        if not paragraphs:
            return None
        
        # Get label if exists
        label = head.get('n')
        
        return Section(
            label=label,
            title=title,
            type=section_type,
            level=1,  # TODO: detect heading levels
            page_start=min(page_nums) if page_nums else 1,
            page_end=max(page_nums) if page_nums else 1,
            paragraphs=paragraphs
        )
    
    def _get_accurate_page_number(self, p_elem: ET.Element, text: str) -> int:
        """Get accurate page number using multiple methods"""
        # Method 1: Try to get from coordinates (GROBID's method)
        coords = p_elem.get('coords')
        if coords:
            try:
                # Parse coords format: "page,x1,y1,x2,y2;..."
                page = int(coords.split(',')[0])
                if page > 0:
                    return page
            except:
                pass
        
        # Method 2: Use PyMuPDF to find page by text content
        if hasattr(self, 'pdf_path') and self.pdf_path.exists():
            try:
                import fitz
                doc = fitz.open(str(self.pdf_path))
                
                # Search for the text in each page
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    # Check if this text appears on this page
                    # Use a more flexible matching approach
                    if self._text_matches_page(text, page_text):
                        doc.close()
                        return page_num + 1
                
                doc.close()
            except Exception as e:
                logger.warning(f"Failed to use PyMuPDF for page detection: {e}")
        
        # Method 3: Fallback to page 1
        return 1
    
    def _text_matches_page(self, search_text: str, page_text: str) -> bool:
        """Check if text content matches a page using flexible matching"""
        # Clean and normalize text
        search_clean = self._normalize_text(search_text)
        page_clean = self._normalize_text(page_text)
        
        # If search text is very short, require exact match
        if len(search_clean) < 20:
            return search_clean in page_clean
        
        # For longer text, use partial matching
        # Split into words and check if most words are present
        search_words = set(search_clean.split())
        page_words = set(page_clean.split())
        
        if not search_words:
            return False
        
        # Calculate word overlap
        common_words = search_words.intersection(page_words)
        overlap_ratio = len(common_words) / len(search_words)
        
        # Require at least 70% word overlap for a match
        return overlap_ratio >= 0.7
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        import re
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase for comparison
        text = text.lower()
        # Remove punctuation for better matching
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _extract_text_from_element(self, elem: ET.Element) -> str:
        """Extract all text from an element and its children"""
        texts = []
        
        if elem.text:
            texts.append(elem.text)
        
        for child in elem:
            if child.tag.endswith('ref'):
                # Handle references
                if child.text:
                    texts.append(child.text)
            else:
                # Recursively extract text
                child_text = self._extract_text_from_element(child)
                if child_text:
                    texts.append(child_text)
            
            if child.tail:
                texts.append(child.tail)
        
        return ' '.join(texts).strip()
    
    def _extract_references(self, root: ET.Element) -> List[Reference]:
        """Extract bibliographic references"""
        references = []
        
        # Find bibliography section
        bibl_struct_list = root.findall('.//tei:listBibl/tei:biblStruct', self.NAMESPACES)
        
        for bibl in bibl_struct_list:
            ref = self._parse_reference(bibl)
            if ref:
                references.append(ref)
        
        return references
    
    def _parse_reference(self, bibl: ET.Element) -> Optional[Reference]:
        """Parse a single bibliographic reference"""
        # Title
        title_elem = bibl.find('.//tei:title', self.NAMESPACES)
        title = title_elem.text if title_elem is not None else None
        
        # Authors
        authors = []
        for author in bibl.findall('.//tei:author', self.NAMESPACES):
            name_parts = []
            forename = author.find('.//tei:forename', self.NAMESPACES)
            surname = author.find('.//tei:surname', self.NAMESPACES)
            
            if forename is not None and forename.text:
                name_parts.append(forename.text)
            if surname is not None and surname.text:
                name_parts.append(surname.text)
            
            if name_parts:
                authors.append(' '.join(name_parts))
        
        # Year
        year = None
        date_elem = bibl.find('.//tei:date', self.NAMESPACES)
        if date_elem is not None:
            when = date_elem.get('when')
            if when:
                try:
                    year = int(when[:4])
                except:
                    pass
        
        # Venue
        venue = None
        meeting = bibl.find('.//tei:meeting', self.NAMESPACES)
        if meeting is not None:
            venue = meeting.text
        else:
            journal = bibl.find('.//tei:title[@level="j"]', self.NAMESPACES)
            if journal is not None:
                venue = journal.text
        
        # DOI
        doi = None
        idno = bibl.find('.//tei:idno[@type="DOI"]', self.NAMESPACES)
        if idno is not None:
            doi = idno.text
        
        # Raw text
        raw_text = self._extract_text_from_element(bibl)
        
        if not raw_text and not title:
            return None
        
        return Reference(
            raw_text=raw_text or "",
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi
        )