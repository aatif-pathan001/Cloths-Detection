"""
FastAPI server entry point.
Run with: uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from app.api.routes import app  # re-export for uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level="info",
    )
