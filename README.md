# LlamaIndex AutoSplit API

A REST API for creating and managing document split jobs using LlamaIndex Cloud.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create a .env file or export the environment variable
export LLAMAINDEX_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Then edit .env and add your API key
```

## Running the API

Start the server:
```bash
# If using .env file, you may need to load it first
# For example, using python-dotenv:
source venv/bin/activate
export LLAMAINDEX_API_KEY="your-api-key-here"
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

- Web interface: `http://localhost:8000`
- API documentation (Swagger UI): `http://localhost:8000/docs`

## API Endpoints

### 1. Create Job (POST `/api/jobs`)

Upload a file to create a new split job.

**Request:**
- Method: `POST`
- URL: `/api/jobs`
- Headers:
  - `X-API-Key`: Your LlamaIndex API key
- Body: `multipart/form-data` with a `file` field

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "message": "Job created successfully"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/jobs" \
  -F "file=@/path/to/your/document.pdf" \
  -F "category_description=Your custom description (optional)"
```

### 2. Get Job Details (GET `/api/jobs/{job_id}`)

Retrieve the status and results of a job.

**Request:**
- Method: `GET`
- URL: `/api/jobs/{job_id}`
- Note: API key is read from `LLAMAINDEX_API_KEY` environment variable

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "created_at": "2024-01-01T12:00:00",
  "result": {
    "segments": [...]
  },
  "error_message": null
}
```

**Example using curl:**
```bash
curl -X GET "http://localhost:8000/api/jobs/{job_id}"
```

## Job Statuses

- `pending`: Job is queued
- `processing`: Job is being processed
- `completed`: Job finished successfully
- `failed`: Job encountered an error

## Notes

- Jobs are stored in-memory. In production, use a database for persistence.
- The API key must be set as the `LLAMAINDEX_API_KEY` environment variable.
- See [LlamaIndex Cloud documentation](https://developers.llamaindex.ai/python/cloud/general/api_key/) for API key setup.
- The web interface is available at `http://localhost:8000` - no API key input needed, it's read from the server environment.

