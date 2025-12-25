from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Optional, List
import requests
import uuid
import time
import io
import base64
import os
from datetime import datetime
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter
import zipfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="LlamaIndex AutoSplit API")

# In-memory storage for jobs (in production, use a database)
jobs_storage = {}

# API configuration
BASE_URL = "https://api.cloud.llamaindex.ai/api/v1"

# Default category description
DEFAULT_CATEGORY_DESCRIPTION = "Pages that belong to the same FACTURA CAMBIARIA invoice but are not the first page. Includes rotated pages, stamps, signatures, Walmart review stamps, or continuation content without the full invoice header."

def get_categories(description: Optional[str] = None) -> List[dict]:
    """Get categories list with optional custom description"""
    return [
        {
            "name": "default",
            "description": description or DEFAULT_CATEGORY_DESCRIPTION
        },
    ]


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobDetails(BaseModel):
    job_id: str
    status: str
    created_at: str
    result: Optional[dict] = None
    error_message: Optional[str] = None


def get_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Get API key from header (if provided) or environment variable"""
    # First check if API key is provided in header (for override)
    if x_api_key:
        return x_api_key
    
    # Otherwise, use environment variable
    api_key = os.getenv("LLAMAINDEX_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="LLAMAINDEX_API_KEY environment variable is not set. Please set it before starting the server or provide X-API-Key header."
        )
    return api_key


@app.post("/api/jobs", response_model=JobResponse)
async def create_job(
    file: UploadFile = File(None),
    file_id: Optional[str] = Form(None),
    category_description: Optional[str] = Form(None),
    api_key: str = Depends(get_api_key)
):
    """
    Create a new split job by uploading a file or using an existing file_id.
    API key is read from LLAMAINDEX_API_KEY environment variable.
    
    Either 'file' (multipart upload) or 'file_id' (if file already uploaded) must be provided.
    
    Optional parameters:
    - category_description: Custom description for the category (default: predefined description)
    """
    # Validate that at least one is provided
    if not file and not file_id:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_id' must be provided")
    
    # Get categories with custom description if provided
    categories = get_categories(category_description)
    
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Prepare headers for LlamaIndex API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        
        # Initialize file_id variable and store original file content
        file_id_from_upload = None
        original_file_content = None
        
        # If file was uploaded, store its content for later PDF splitting
        if file:
            await file.seek(0)
            original_file_content = await file.read()
            await file.seek(0)  # Reset for potential upload
        
        # If file_id is provided, skip upload step
        if file_id:
            # Validate file_id is a valid UUID format
            try:
                uuid.UUID(file_id)
                file_id_from_upload = file_id
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid file_id format. Must be a valid UUID. Got: {file_id}")
        elif file:
            # Step 1: Upload file to LlamaIndex to get file_id
            # Reset file pointer to beginning
            await file.seek(0)
            file_content = await file.read()
        
            upload_headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
            
            # Try multipart upload with different field names and endpoints
            # Based on LlamaIndex API structure, the endpoint is likely /v1/files
            upload_response = None
            field_names_to_try = ["file", "document", "upload", "data", "upload_file"]
            endpoints_to_try_first = [
                f"{BASE_URL}/files",  # /api/v1/files (most likely)
                f"https://api.cloud.llamaindex.ai/api/v1/files",  # Explicit full path
                f"{BASE_URL}/beta/files",  # Fallback to beta
            ]
        
            for endpoint in endpoints_to_try_first:
                for field_name in field_names_to_try:
                    # Try with just filename and content
                    files_data = {
                        field_name: (file.filename, file_content, file.content_type or "application/pdf")
                    }
                    
                    upload_response = requests.post(
                        endpoint,
                        headers=upload_headers,
                        files=files_data
                    )
                    
                    # If successful, break out of loop
                    if upload_response.ok:
                        break
                    
                    # If 422 error, try with additional metadata
                    if upload_response.status_code == 422:
                        # Try with metadata fields
                        files_data_with_meta = {
                            field_name: (file.filename, file_content, file.content_type or "application/pdf"),
                            "filename": (None, file.filename),
                        }
                        upload_response = requests.post(
                            endpoint,
                            headers=upload_headers,
                            files=files_data_with_meta
                        )
                        if upload_response.ok:
                            break
                
                if upload_response and upload_response.ok:
                    break
            
            # If all multipart attempts failed, try different JSON formats and endpoints
            # Check if we got any response at all
            if not upload_response or (upload_response and upload_response.status_code == 422):
                file_base64 = base64.b64encode(file_content).decode('utf-8')
                upload_headers_json = {
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
                
                # Try different JSON payload structures
                json_payloads_to_try = [
                    {"file": file_base64, "filename": file.filename},
                    {"data": file_base64, "filename": file.filename},
                    {"content": file_base64, "filename": file.filename, "content_type": file.content_type or "application/pdf"},
                    {"file_data": file_base64, "name": file.filename},
                    {"document": file_base64, "filename": file.filename},
                ]
                
                # Try different endpoints (JSON format)
                endpoints_to_try = [
                    f"{BASE_URL}/files",  # /v1/files
                    f"{BASE_URL}/beta/files",
                    f"https://api.cloud.llamaindex.ai/api/v1/files",
                ]
                
                upload_response = None
                for endpoint in endpoints_to_try:
                    for json_payload in json_payloads_to_try:
                        upload_response = requests.post(
                            endpoint,
                            headers=upload_headers_json,
                            json=json_payload
                        )
                        if upload_response.ok:
                            break
                    if upload_response and upload_response.ok:
                        break
                
                # If JSON formats failed, try raw binary with different endpoints
                if not upload_response or not upload_response.ok:
                    upload_headers_binary = {
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                        "Content-Type": file.content_type or "application/pdf",
                    }
                    for endpoint in endpoints_to_try:
                        upload_response = requests.post(
                            endpoint,
                            headers=upload_headers_binary,
                            data=file_content
                        )
                        if upload_response.ok:
                            break
            
            # Check if file upload succeeded
            if upload_response and upload_response.ok:
                try:
                    file_data = upload_response.json()
                    # Try different possible keys for file_id
                    file_id_from_upload = (
                        file_data.get("id") or 
                        file_data.get("file_id") or 
                        file_data.get("fileId") or
                        file_data.get("uuid")
                    )
                    
                    if not file_id_from_upload:
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Failed to get file_id from upload response. Response keys: {list(file_data.keys())}. Full response: {file_data}"
                        )
                    
                    # Validate it's a valid UUID
                    try:
                        uuid.UUID(str(file_id_from_upload))
                    except ValueError:
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Invalid file_id format from upload (not a valid UUID): {file_id_from_upload}. Response: {file_data}"
                        )
                except ValueError as e:
                    raise HTTPException(status_code=500, detail=f"Error parsing upload response: {str(e)}")
            
            # If file upload failed, try sending file directly in split job creation
            if not file_id_from_upload:
                # Try creating split job with file directly (some APIs support this)
                file_base64 = base64.b64encode(file_content).decode('utf-8')
                
                # Try with document_input type "file" or "base64" instead of "file_id"
                split_response = None
                document_inputs_to_try = [
                    {"type": "file", "value": file_base64, "filename": file.filename},
                    {"type": "base64", "value": file_base64, "filename": file.filename},
                    {"type": "file_data", "value": file_base64, "filename": file.filename},
                ]
                
                for doc_input in document_inputs_to_try:
                    try:
                        split_response = requests.post(
                            f"{BASE_URL}/beta/split/jobs",
                            headers={
                                **headers,
                                "Content-Type": "application/json"
                            },
                            json={
                                "document_input": doc_input,
                                "categories": categories,
                                "splitting_strategy": {
                                    "allow_uncategorized": False,
                                },
                            },
                        )
                        if split_response.ok:
                            break
                    except:
                        continue
                
                # If direct file upload in job creation worked, skip file upload step
                if split_response and split_response.ok:
                    split_job = split_response.json()
                    file_id_from_upload = "direct_upload"  # Placeholder since we didn't upload separately
                else:
                    # File upload failed and direct job creation failed - return detailed error
                    error_detail = "Unknown error"
                    status_code = 500
                    if upload_response:
                        status_code = upload_response.status_code
                        try:
                            error_text = upload_response.text
                            error_json = upload_response.json()
                            error_detail = error_json.get("detail", error_text)
                            if isinstance(error_detail, list) and len(error_detail) > 0:
                                error_detail = error_detail[0].get("msg", str(error_detail))
                        except:
                            error_detail = upload_response.text[:500] if upload_response.text else "No error details"
                    else:
                        error_detail = "No response from file upload endpoint. Tried endpoints: /v1/files, /beta/files with multiple formats (multipart, JSON, binary)."
                    
                    raise HTTPException(
                        status_code=status_code,
                        detail=f"File upload failed. Error: {error_detail}. "
                               f"Note: LlamaIndex file upload endpoint may require a different format or authentication. "
                               f"Alternative: Upload the file separately using LlamaIndex SDK/API and provide the file_id directly in the request."
                    )
        
        # Step 2: Create split job with file_id
        # If file_id was provided directly or we got it from upload, create split job
        if file_id_from_upload and file_id_from_upload != "direct_upload":
            # Validate file_id is a valid UUID before using
            try:
                validated_uuid = uuid.UUID(str(file_id_from_upload))
                file_id_str = str(validated_uuid)  # Ensure it's a string in UUID format
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file_id format: {file_id_from_upload}. Must be a valid UUID. Please ensure the file was uploaded successfully."
                )
            
            split_response = requests.post(
                f"{BASE_URL}/beta/split/jobs",
                headers={
                    **headers,
                    "Content-Type": "application/json"
                },
                json={
                    "document_input": {
                        "type": "file_id",
                        "value": file_id_str,  # Use the validated UUID string
                    },
                    "categories": categories,
                    "splitting_strategy": {
                        "allow_uncategorized": False,
                    },
                },
            )
            
            if not split_response.ok:
                error_text = split_response.text
                try:
                    error_json = split_response.json()
                    error_detail = error_json.get("detail", error_text)
                except:
                    error_detail = error_text
                raise HTTPException(
                    status_code=split_response.status_code,
                    detail=f"Failed to create split job: {error_detail}"
                )
            
            split_job = split_response.json()
        elif file_id_from_upload == "direct_upload":
            # split_job was already created in the direct upload attempt above
            pass
        else:
            raise HTTPException(status_code=500, detail="Failed to get file_id for split job creation")
        
        # Store job information
        jobs_storage[job_id] = {
            "job_id": job_id,
            "llamaindex_job_id": split_job.get("id"),
            "status": split_job.get("status", "pending"),
            "created_at": datetime.now().isoformat(),
            "file_id": file_id_from_upload if file_id_from_upload != "direct_upload" else None,
            "filename": file.filename if file else None,
            "original_file_content": original_file_content,  # Store for PDF splitting
            "result": None,
            "error_message": None,
        }
        
        return JobResponse(
            job_id=job_id,
            status=jobs_storage[job_id]["status"],
            message="Job created successfully"
        )
        
    except requests.exceptions.HTTPError as e:
        error_detail = f"LlamaIndex API error: {e.response.text if hasattr(e, 'response') else str(e)}"
        raise HTTPException(status_code=e.response.status_code if hasattr(e, 'response') else 500, detail=error_detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/jobs/{job_id}", response_model=JobDetails)
async def get_job_details(
    job_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Get job details by job ID.
    API key is read from LLAMAINDEX_API_KEY environment variable.
    """
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    
    # Prepare headers for LlamaIndex API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    # Fetch latest status from LlamaIndex API
    try:
        llamaindex_job_id = job.get("llamaindex_job_id")
        if llamaindex_job_id:
            response = requests.get(
                f"{BASE_URL}/beta/split/jobs/{llamaindex_job_id}",
                headers=headers,
            )
            response.raise_for_status()
            llamaindex_job = response.json()
            
            # Update job status
            job["status"] = llamaindex_job.get("status", job["status"])
            
            if llamaindex_job.get("status") == "completed" and llamaindex_job.get("result"):
                job["result"] = llamaindex_job.get("result")
            elif llamaindex_job.get("status") == "failed":
                job["error_message"] = llamaindex_job.get("error_message", "Job failed")
            
            # Update storage
            jobs_storage[job_id] = job
    except requests.exceptions.HTTPError as e:
        # If API call fails, return stored job info
        pass
    except Exception as e:
        # If any other error, return stored job info
        pass
    
    return JobDetails(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        result=job.get("result"),
        error_message=job.get("error_message")
    )


@app.post("/api/jobs/{job_id}/split-pdf")
async def split_pdf_by_confidence(
    job_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Split the PDF based on high confidence pages.
    Returns a ZIP file containing split PDFs.
    API key is read from LLAMAINDEX_API_KEY environment variable.
    """
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    
    # Check if job is completed
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed. Current status: {job.get('status')}")
    
    # Check if result is available
    result = job.get("result")
    if not result or "segments" not in result:
        raise HTTPException(status_code=400, detail="Job result not available or invalid")
    
    # Check if original file is stored
    original_file_content = job.get("original_file_content")
    if not original_file_content:
        raise HTTPException(status_code=400, detail="Original file not available. Please upload file again or use file_id method.")
    
    # Find pages with high confidence
    segments = result["segments"]
    high_confidence_pages = []
    
    for segment in segments:
        if segment.get("confidence_category") == "high":
            pages = segment.get("pages", [])
            if isinstance(pages, list) and len(pages) > 0:
                high_confidence_pages.extend(pages)
    
    if not high_confidence_pages:
        raise HTTPException(status_code=400, detail="No high confidence pages found in the result")
    
    # Sort high confidence pages
    high_confidence_pages = sorted(set(high_confidence_pages))
    
    try:
        # Read the original PDF
        pdf_reader = PdfReader(io.BytesIO(original_file_content))
        total_pages = len(pdf_reader.pages)
        
        # Create split points based on high confidence pages
        # Logic: Split before first high confidence, then at each high confidence page
        split_ranges = []
        
        # First PDF: pages 1 to (first high confidence - 1)
        if high_confidence_pages[0] > 1:
            split_ranges.append((1, high_confidence_pages[0] - 1))
        
        # Create ranges starting from each high confidence page
        for i, high_page in enumerate(high_confidence_pages):
            start_page = high_page
            # End at next high confidence page - 1, or end of document
            if i + 1 < len(high_confidence_pages):
                end_page = high_confidence_pages[i + 1] - 1
            else:
                end_page = total_pages
            
            split_ranges.append((start_page, end_page))
        
        # Create split PDFs
        split_pdfs = []
        for idx, (start, end) in enumerate(split_ranges, 1):
            writer = PdfWriter()
            
            # Add pages (PDF pages are 0-indexed)
            for page_num in range(start - 1, min(end, total_pages)):
                writer.add_page(pdf_reader.pages[page_num])
            
            # Create PDF in memory
            pdf_buffer = io.BytesIO()
            writer.write(pdf_buffer)
            pdf_buffer.seek(0)
            
            original_filename = job.get("filename", "document.pdf")
            base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
            split_filename = f"{base_name}_part_{idx}_pages_{start}-{end}.pdf"
            
            split_pdfs.append({
                "filename": split_filename,
                "content": pdf_buffer.getvalue(),
                "pages": f"{start}-{end}"
            })
        
        # Create ZIP file with all split PDFs
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for split_pdf in split_pdfs:
                zip_file.writestr(split_pdf["filename"], split_pdf["content"])
        
        zip_buffer.seek(0)
        
        # Return ZIP file
        original_filename = job.get("filename", "document.pdf")
        base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
        zip_filename = f"{base_name}_split_pdfs.zip"
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.getvalue()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error splitting PDF: {str(e)}")


@app.get("/")
async def root():
    """Serve the web interface"""
    import os
    # Serve the HTML file if it exists
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "LlamaIndex AutoSplit API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"message": "LlamaIndex AutoSplit API", "status": "running"}

