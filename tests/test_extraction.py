"""
test_extraction.py - Test script for PDF extraction

Usage:
    python test_extraction.py [pdf_path]
    
If no PDF path is provided, it will look for a PDF in the paper folder.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from services.pipeline import ExtractionPipeline
from models.schemas import ExtractionRequest
from utils.helpers import validate_pdf, get_pdf_info, create_extraction_summary
from loguru import logger


async def test_extraction(pdf_path: Optional[Path] = None):
    """
    Test PDF extraction with all methods
    """
    print("=" * 60)
    print("PDF Extraction Test")
    print("=" * 60)
    
    # Find PDF to process
    if pdf_path and pdf_path.exists():
        target_pdf = pdf_path
    else:
        # Look in paper folder
        pdf_files = list(settings.paper_folder.glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF found in paper folder")
            print(f"   Please place a PDF in: {settings.paper_folder}")
            return
        
        if len(pdf_files) > 1:
            print("⚠️  Multiple PDFs found:")
            for i, pdf in enumerate(pdf_files):
                print(f"   {i+1}. {pdf.name}")
            choice = input("Select PDF number (default=1): ").strip()
            idx = int(choice) - 1 if choice.isdigit() else 0
            target_pdf = pdf_files[idx]
        else:
            target_pdf = pdf_files[0]
    
    print(f"\n📄 Processing: {target_pdf.name}")
    
    # Validate PDF
    print("\n🔍 Validating PDF...")
    if not validate_pdf(target_pdf):
        print("❌ Invalid PDF file")
        return
    
    # Get PDF info
    pdf_info = get_pdf_info(target_pdf)
    print(f"   Pages: {pdf_info['page_count']}")
    print(f"   Size: {pdf_info['file_size'] / 1024 / 1024:.2f} MB")
    print(f"   Has Text: {pdf_info['has_text']}")
    print(f"   Has Images: {pdf_info['has_images']}")
    print(f"   Encrypted: {pdf_info['encrypted']}")
    
    if pdf_info['metadata'].get('title'):
        print(f"   Title: {pdf_info['metadata']['title']}")
    
    # Initialize pipeline
    print("\n🚀 Initializing extraction pipeline...")
    pipeline = ExtractionPipeline()
    
    # Create extraction request
    request = ExtractionRequest(
        pdf_path=str(target_pdf),
        extract_text=True,
        extract_figures=True,
        extract_tables=True,
        extract_equations=True,
        extract_code=True,
        extract_references=True,
        use_ocr=True,
        detect_entities=True
    )
    
    # Run extraction
    print("\n⚙️  Starting extraction...")
    print("   This may take a few minutes depending on PDF size...")
    
    start_time = datetime.now()
    
    try:
        result = await pipeline.extract(target_pdf, request)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ Extraction completed in {elapsed_time:.2f} seconds")
        
        # Print results summary
        print("\n📊 Extraction Results:")
        print(f"   Status: {result.status}")
        print(f"   Coverage: {result.extraction_coverage}%")
        print(f"   Methods Used: {', '.join(result.extraction_methods)}")
        
        print("\n📝 Content Extracted:")
        print(f"   Sections: {len(result.sections)}")
        print(f"   Figures: {len(result.figures)}")
        print(f"   Tables: {len(result.tables)}")
        print(f"   Equations: {len(result.equations)}")
        print(f"   Code Blocks: {len(result.code_blocks)}")
        print(f"   References: {len(result.references)}")
        print(f"   Entities: {len(result.entities)}")
        
        # Show confidence scores
        if result.confidence_scores:
            print("\n🎯 Confidence Scores:")
            for key, score in result.confidence_scores.items():
                print(f"   {key.capitalize()}: {score:.2f}")
        
        # Show any errors or warnings
        if result.errors:
            print("\n⚠️  Errors:")
            for error in result.errors:
                print(f"   - {error}")
        
        if result.warnings:
            print("\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")
        
        # Save results
        output_path = settings.paper_folder / f"{target_pdf.stem}_extraction.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Results saved to: {output_path}")
        
        # Create and save summary
        summary = create_extraction_summary(result.dict())
        summary_path = settings.paper_folder / f"{target_pdf.stem}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"📄 Summary saved to: {summary_path}")
        
        # Show sample extracted content
        print("\n📖 Sample Extracted Content:")
        
        # Show title and abstract
        if result.metadata.title != "Unknown":
            print(f"\nTitle: {result.metadata.title}")
        
        if result.metadata.abstract:
            print(f"\nAbstract: {result.metadata.abstract[:200]}...")
        
        # Show first section
        if result.sections:
            first_section = result.sections[0]
            print(f"\nFirst Section: {first_section.title}")
            if first_section.paragraphs:
                first_para = first_section.paragraphs[0]
                text = first_para.text if hasattr(first_para, 'text') else first_para.get('text', '')
                print(f"   {text[:200]}...")
        
        # Show first figure
        if result.figures:
            first_figure = result.figures[0]
            print(f"\nFirst Figure: {first_figure.label}")
            if first_figure.caption:
                print(f"   Caption: {first_figure.caption[:100]}...")
        
        # Show first table
        if result.tables:
            first_table = result.tables[0]
            print(f"\nFirst Table: {first_table.label}")
            if first_table.caption:
                print(f"   Caption: {first_table.caption[:100]}...")
            if first_table.headers:
                print(f"   Headers: {first_table.headers[0][:5]}...")
        
        # Show first equation
        if result.equations:
            first_eq = result.equations[0]
            print(f"\nFirst Equation: {first_eq.latex[:50]}...")
        
        # Show first code block
        if result.code_blocks:
            first_code = result.code_blocks[0]
            print(f"\nFirst Code Block ({first_code.language or 'unknown'}):")
            print(f"   {first_code.code[:100]}...")
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    """Main entry point"""
    # Parse command line arguments
    pdf_path = None
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            sys.exit(1)
    
    # Run test
    asyncio.run(test_extraction(pdf_path))


if __name__ == "__main__":
    main()