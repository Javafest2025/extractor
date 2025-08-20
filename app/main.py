# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List
import uuid
import shutil
from datetime import datetime
from loguru import logger

from app.config import settings
from app.models.schemas import (
    ExtractionRequest, ExtractionResponse, ExtractionResult,
    ExtractionStatus
)
from app.services.pipeline import ExtractionPipeline
from app.utils.helpers import validate_pdf, get_pdf_info, create_extraction_summary
from app.utils.exceptions import InvalidPDFError, ExtractionError

# Configure logging
logger.add(
    "logs/pdf_extractor.log",
    rotation="500 MB",
    retention="10 days",
    level=settings.log_level
)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Advanced PDF Extraction API for Academic Papers",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize extraction pipeline
pipeline = ExtractionPipeline()

# In-memory storage for job tracking (use Redis in production)
extraction_jobs = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Create necessary directories
    settings.paper_folder.mkdir(exist_ok=True)
    (settings.paper_folder / "uploads").mkdir(exist_ok=True)
    (settings.paper_folder / "results").mkdir(exist_ok=True)
    
    logger.info("Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "api_docs": f"{settings.api_prefix}/docs"
    }


@app.get(f"{settings.api_prefix}/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version
    }


@app.post(f"{settings.api_prefix}/extract")
async def extract_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_text: bool = Query(True, description="Extract text and structure"),
    extract_figures: bool = Query(True, description="Extract figures"),
    extract_tables: bool = Query(True, description="Extract tables"),
    extract_equations: bool = Query(True, description="Extract equations"),
    extract_code: bool = Query(True, description="Extract code blocks"),
    extract_references: bool = Query(True, description="Extract references"),
    use_ocr: bool = Query(True, description="Use OCR for scanned PDFs"),
    detect_entities: bool = Query(True, description="Detect named entities"),
    async_processing: bool = Query(False, description="Process asynchronously")
):
    """
    Extract content from uploaded PDF
    
    This endpoint accepts a PDF file and extracts various types of content
    using multiple extraction techniques for maximum accuracy and coverage.
    """
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = settings.paper_folder / "uploads" / f"{job_id}.pdf"
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Validate PDF
    if not validate_pdf(upload_path):
        upload_path.unlink()  # Delete invalid file
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file")
    
    # Create extraction request
    request = ExtractionRequest(
        pdf_path=str(upload_path),
        extract_text=extract_text,
        extract_figures=extract_figures,
        extract_tables=extract_tables,
        extract_equations=extract_equations,
        extract_code=extract_code,
        extract_references=extract_references,
        use_ocr=use_ocr,
        detect_entities=detect_entities
    )
    
    # Initialize job tracking
    extraction_jobs[job_id] = {
        "status": ExtractionStatus.PENDING,
        "created_at": datetime.utcnow(),
        "file_name": file.filename,
        "file_path": str(upload_path)
    }
    
    if async_processing:
        # Process in background
        background_tasks.add_task(process_extraction, job_id, upload_path, request)
        
        return ExtractionResponse(
            job_id=job_id,
            status=ExtractionStatus.PENDING,
            message="Extraction job started. Use /status endpoint to check progress."
        )
    else:
        # Process synchronously
        try:
            result = await pipeline.extract(upload_path, request)
            
            # Update job status
            extraction_jobs[job_id]["status"] = result.status
            extraction_jobs[job_id]["result"] = result
            extraction_jobs[job_id]["completed_at"] = datetime.utcnow()
            
            return ExtractionResponse(
                job_id=job_id,
                status=result.status,
                result=result,
                message="Extraction completed successfully"
            )
        except Exception as e:
            logger.error(f"Extraction failed for job {job_id}: {e}")
            extraction_jobs[job_id]["status"] = ExtractionStatus.FAILED
            extraction_jobs[job_id]["error"] = str(e)
            
            raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


async def process_extraction(job_id: str, pdf_path: Path, request: ExtractionRequest):
    """Background task for PDF extraction"""
    try:
        extraction_jobs[job_id]["status"] = ExtractionStatus.PROCESSING
        extraction_jobs[job_id]["started_at"] = datetime.utcnow()
        
        result = await pipeline.extract(pdf_path, request)
        
        extraction_jobs[job_id]["status"] = result.status
        extraction_jobs[job_id]["result"] = result
        extraction_jobs[job_id]["completed_at"] = datetime.utcnow()
        
        logger.info(f"Extraction completed for job {job_id}")
    except Exception as e:
        logger.error(f"Extraction failed for job {job_id}: {e}")
        extraction_jobs[job_id]["status"] = ExtractionStatus.FAILED
        extraction_jobs[job_id]["error"] = str(e)
        extraction_jobs[job_id]["completed_at"] = datetime.utcnow()


@app.get(f"{settings.api_prefix}/status/{{job_id}}")
async def get_job_status(job_id: str):
    """Get status of extraction job"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = extraction_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "file_name": job.get("file_name"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at")
    }
    
    if job["status"] == ExtractionStatus.COMPLETED:
        response["result"] = job.get("result")
    elif job["status"] == ExtractionStatus.FAILED:
        response["error"] = job.get("error")
    elif job["status"] == ExtractionStatus.PROCESSING:
        # Estimate progress (simplified)
        if "started_at" in job:
            elapsed = (datetime.utcnow() - job["started_at"]).total_seconds()
            estimated_total = settings.extraction_timeout
            progress = min(95, (elapsed / estimated_total) * 100)
            response["progress"] = progress
    
    return response


@app.get(f"{settings.api_prefix}/result/{{job_id}}")
async def get_extraction_result(job_id: str, format: str = Query("json", enum=["json", "summary"])):
    """Get extraction result"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = extraction_jobs[job_id]
    
    if job["status"] != ExtractionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job['status']}. Results only available for completed jobs."
        )
    
    result = job.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    if format == "summary":
        # Return human-readable summary
        summary = create_extraction_summary(result.dict())
        return {"summary": summary}
    else:
        # Return full JSON result
        return result


@app.get(f"{settings.api_prefix}/download/{{job_id}}")
async def download_result(job_id: str):
    """Download extraction result as JSON file"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = extraction_jobs[job_id]
    
    if job["status"] != ExtractionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job['status']}. Download only available for completed jobs."
        )
    
    # Check if result file exists
    result_path = settings.paper_folder / "results" / f"{job_id}_extraction.json"
    
    if not result_path.exists():
        # Save result to file
        result = job.get("result")
        if result:
            import json
            with open(result_path, 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)
        else:
            raise HTTPException(status_code=404, detail="Result not found")
    
    return FileResponse(
        path=result_path,
        media_type="application/json",
        filename=f"{job['file_name']}_extraction.json"
    )


@app.post(f"{settings.api_prefix}/extract-from-folder")
async def extract_from_paper_folder():
    """
    Extract content from PDF in the paper folder
    
    This endpoint processes the single PDF file in the paper folder
    as specified in the requirements.
    """
    # Find PDF in paper folder
    pdf_files = list(settings.paper_folder.glob("*.pdf"))
    
    if not pdf_files:
        raise HTTPException(status_code=404, detail="No PDF found in paper folder")
    
    if len(pdf_files) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Multiple PDFs found in paper folder. Expected only one. Found: {[f.name for f in pdf_files]}"
        )
    
    pdf_path = pdf_files[0]
    
    # Validate PDF
    if not validate_pdf(pdf_path):
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {pdf_path.name}")
    
    # Get PDF info
    pdf_info = get_pdf_info(pdf_path)
    
    logger.info(f"Processing PDF: {pdf_path.name}")
    logger.info(f"PDF Info: {pdf_info}")
    
    # Create extraction request with all features enabled
    request = ExtractionRequest(
        pdf_path=str(pdf_path),
        extract_text=True,
        extract_figures=True,
        extract_tables=True,
        extract_equations=True,
        extract_code=True,
        extract_references=True,
        use_ocr=True,
        detect_entities=True
    )
    
    try:
        # Run extraction
        result = await pipeline.extract(pdf_path, request)
        
        # Save result to JSON in the same folder
        output_path = settings.paper_folder / f"{pdf_path.stem}_extraction.json"
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False, default=str)
        
        # Create summary
        summary = create_extraction_summary(result.dict())
        
        # Save summary
        summary_path = settings.paper_folder / f"{pdf_path.stem}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Extraction completed. Results saved to {output_path}")
        logger.info(f"Summary saved to {summary_path}")
        
        return {
            "status": "success",
            "pdf_file": pdf_path.name,
            "pdf_info": pdf_info,
            "extraction_result": result,
            "output_files": {
                "json": str(output_path),
                "summary": str(summary_path)
            },
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.delete(f"{settings.api_prefix}/cleanup")
async def cleanup_old_jobs(hours: int = Query(24, description="Delete jobs older than N hours")):
    """Clean up old extraction jobs and files"""
    from datetime import timedelta
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    deleted_count = 0
    
    jobs_to_delete = []
    for job_id, job in extraction_jobs.items():
        created_at = job.get("created_at")
        if created_at and created_at < cutoff_time:
            jobs_to_delete.append(job_id)
            
            # Delete associated files
            if "file_path" in job:
                try:
                    Path(job["file_path"]).unlink(missing_ok=True)
                except:
                    pass
    
    for job_id in jobs_to_delete:
        del extraction_jobs[job_id]
        deleted_count += 1
    
    return {
        "deleted_jobs": deleted_count,
        "remaining_jobs": len(extraction_jobs)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )