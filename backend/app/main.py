from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from app.routes.generate_report import router as report_router
from app.services.audio_service import get_audio_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting Clinical Documentation Assistant Backend")
    logger.info("Python 3.13.3 with Med-Gemma AI Model")
    
    # Verify audio service
    audio_service = get_audio_service()
    logger.info(f"✅ Audio Service initialized with {len(audio_service.get_supported_languages())} supported languages")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Clinical Documentation Assistant Backend")


# Create FastAPI application
app = FastAPI(
    title="Clinical Documentation Assistant API",
    description="AI-powered medical report generation from audio recordings using Med-Gemma",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "medical-reports",
            "description": "Generate medical reports from audio/text using Med-Gemma AI"
        }
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"  # For demo purposes
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routers
app.include_router(report_router)

# Add middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"📤 {request.method} {request.url.path} "
        f"completed in {process_time:.3f}s "
        f"status={response.status_code}"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Clinical Documentation Assistant",
        "version": "2.0.0",
        "ai_model": "Med-Gemma (google/medgemma-2b)",
        "description": "AI-powered medical report generation from doctor-patient conversations",
        "endpoints": {
            "generate_from_text": "POST /api/v1/generate_report",
            "generate_from_audio": "POST /api/v1/generate_report/audio",
            "generate_from_base64": "POST /api/v1/generate_report/audio_base64",
            "transcribe_only": "POST /api/v1/transcribe_audio",
            "health_check": "GET /api/v1/health",
            "report_types": "GET /api/v1/report_types",
            "audio_formats": "GET /api/v1/supported_audio_formats",
            "languages": "GET /api/v1/supported_languages",
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        },
        "features": [
            "Audio-to-text transcription",
            "8+ medical report types",
            "Med-Gemma AI integration",
            "Multi-language support",
            "Real-time processing"
        ]
    }


# Health endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "clinical-documentation-assistant",
        "version": "2.0.0",
        "timestamp": time.time()
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": request.url.path,
            "suggestion": "Check available endpoints at /"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc) if str(exc) else "Unknown error",
            "support": "Check logs for more details"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1  # Single worker for model memory efficiency
    )