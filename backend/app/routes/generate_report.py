from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from typing import Optional, Dict, List
import logging
import base64
import time

from app.schemas.report import (
    ReportRequest, ReportResponse, AudioReportRequest,
    TranscriptionResult, ReportType, AudioFormat, HealthResponse
)
from app.services.ai_service import get_medgemma_service
from app.services.audio_service import get_audio_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["medical-reports"])

# Initialize services
medgemma_service = get_medgemma_service()
audio_service = get_audio_service()


@router.on_event("startup")
async def startup_event():
    """Initialize services when the app starts"""
    logger.info("Initializing Med-Gemma service...")
    medgemma_service.initialize()
    logger.info("Services initialized successfully")


@router.post("/generate_report", response_model=ReportResponse)
async def generate_report_from_text(
    request: ReportRequest,
    background_tasks: BackgroundTasks = None
) -> ReportResponse:
    """
    Generate a medical report from clinical notes text
    
    Args:
        request: ReportRequest containing clinical notes and parameters
        background_tasks: FastAPI background tasks (optional)
        
    Returns:
        ReportResponse with generated medical report
    """
    start_time = time.time()
    
    try:
        logger.info(f"Generating {request.report_type.value} report from text input")
        
        # Prepare patient context
        patient_context = {}
        if request.patient_age:
            patient_context["age"] = request.patient_age
        if request.patient_gender:
            patient_context["gender"] = request.patient_gender
        
        # Generate report using Med-Gemma
        generation_result = medgemma_service.generate_medical_report(
            clinical_notes=request.clinical_notes,
            report_type=request.report_type.value,
            patient_context=patient_context,
            max_new_tokens=1024
        )
        
        if not generation_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail="Failed to generate medical report"
            )
        
        # Create response
        response = ReportResponse(
            report=generation_result["report"],
            sections=generation_result.get("sections", {}),
            metadata={
                "report_type": request.report_type.value,
                "language": request.language,
                "model_used": generation_result.get("model", "medgemma"),
                "is_mock": generation_result.get("is_mock", False),
                "input_length": len(request.clinical_notes),
                "output_length": len(generation_result["report"]),
                "processing_time": time.time() - start_time
            },
            model_used=generation_result.get("model", "medgemma"),
            report_type=request.report_type.value
        )
        
        logger.info(f"Successfully generated report in {time.time() - start_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/generate_report/audio", response_model=ReportResponse)
async def generate_report_from_audio(
    audio_file: UploadFile = File(..., description="Audio recording of doctor-patient conversation"),
    report_type: str = Form(ReportType.CLINICAL_REPORT.value, description="Type of medical report"),
    language: str = Form("en-US", description="Language for speech recognition"),
    patient_age: Optional[int] = Form(None, description="Patient age"),
    patient_gender: Optional[str] = Form(None, description="Patient gender"),
    background_tasks: BackgroundTasks = None
) -> ReportResponse:
    """
    Generate a medical report from audio recording
    
    Args:
        audio_file: Audio file containing doctor-patient conversation
        report_type: Type of medical report to generate
        language: Language code for speech recognition
        patient_age: Optional patient age
        patient_gender: Optional patient gender
        
    Returns:
        ReportResponse with generated medical report and transcription
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing audio file: {audio_file.filename}")
        
        # Read audio file
        audio_data = await audio_file.read()
        
        # Validate audio format from filename
        audio_format = audio_file.filename.split('.')[-1].lower()
        if audio_format not in [fmt.value for fmt in AudioFormat]:
            audio_format = "wav"  # Default
        
        # Validate audio
        is_valid, error_msg = audio_service.validate_audio(audio_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid audio: {error_msg}")
        
        # Convert audio to text
        transcription_start = time.time()
        transcribed_text, confidence = audio_service.audio_to_text(
            audio_data, 
            audio_format=audio_format,
            language=language
        )
        transcription_time = time.time() - transcription_start
        
        if not transcribed_text:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio. Please ensure clear speech and try again."
            )
        
        logger.info(f"Audio transcribed in {transcription_time:.2f}s (confidence: {confidence or 0:.2f})")
        
        # Prepare patient context
        patient_context = {}
        if patient_age:
            patient_context["age"] = patient_age
        if patient_gender:
            patient_context["gender"] = patient_gender
        
        # Generate report using Med-Gemma
        generation_start = time.time()
        generation_result = medgemma_service.generate_medical_report(
            clinical_notes=transcribed_text,
            report_type=report_type,
            patient_context=patient_context,
            max_new_tokens=1024
        )
        generation_time = time.time() - generation_start
        
        if not generation_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail="Failed to generate medical report from transcription"
            )
        
        # Create transcription result
        transcription_result = TranscriptionResult(
            text=transcribed_text,
            confidence=confidence or 0.5,
            language=language
        )
        
        # Create response
        response = ReportResponse(
            report=generation_result["report"],
            sections=generation_result.get("sections", {}),
            transcription=transcription_result,
            metadata={
                "report_type": report_type,
                "language": language,
                "model_used": generation_result.get("model", "medgemma"),
                "is_mock": generation_result.get("is_mock", False),
                "audio_format": audio_format,
                "audio_size_bytes": len(audio_data),
                "transcription_confidence": confidence,
                "transcription_time": transcription_time,
                "generation_time": generation_time,
                "total_processing_time": time.time() - start_time
            },
            model_used=generation_result.get("model", "medgemma"),
            report_type=report_type
        )
        
        logger.info(f"Successfully processed audio and generated report in {time.time() - start_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio and generating report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/generate_report/audio_base64", response_model=ReportResponse)
async def generate_report_from_audio_base64(
    request: AudioReportRequest,
    background_tasks: BackgroundTasks = None
) -> ReportResponse:
    """
    Generate a medical report from base64 encoded audio
    
    Args:
        request: AudioReportRequest with base64 audio and parameters
        
    Returns:
        ReportResponse with generated medical report
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing base64 audio for {request.report_type.value} report")
        
        # Decode base64 audio
        try:
            audio_data = base64.b64decode(request.audio_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {str(e)}")
        
        # Validate audio
        is_valid, error_msg = audio_service.validate_audio(audio_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid audio: {error_msg}")
        
        # Convert audio to text
        transcribed_text, confidence = audio_service.audio_to_text(
            audio_data,
            audio_format=request.audio_format.value,
            language=request.language
        )
        
        if not transcribed_text:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio. Please ensure clear speech and try again."
            )
        
        logger.info(f"Audio transcribed (confidence: {confidence or 0:.2f})")
        
        # Prepare patient context
        patient_context = {}
        if request.patient_age:
            patient_context["age"] = request.patient_age
        if request.patient_gender:
            patient_context["gender"] = request.patient_gender
        
        # Generate report using Med-Gemma
        generation_result = medgemma_service.generate_medical_report(
            clinical_notes=transcribed_text,
            report_type=request.report_type.value,
            patient_context=patient_context,
            max_new_tokens=1024
        )
        
        if not generation_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail="Failed to generate medical report from transcription"
            )
        
        # Create transcription result
        transcription_result = TranscriptionResult(
            text=transcribed_text,
            confidence=confidence or 0.5,
            language=request.language
        )
        
        # Create response
        response = ReportResponse(
            report=generation_result["report"],
            sections=generation_result.get("sections", {}),
            transcription=transcription_result,
            metadata={
                "report_type": request.report_type.value,
                "language": request.language,
                "model_used": generation_result.get("model", "medgemma"),
                "is_mock": generation_result.get("is_mock", False),
                "audio_format": request.audio_format.value,
                "audio_size_bytes": len(audio_data),
                "transcription_confidence": confidence
            },
            model_used=generation_result.get("model", "medgemma"),
            report_type=request.report_type.value
        )
        
        logger.info(f"Successfully processed base64 audio and generated report in {time.time() - start_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing base64 audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    ai_health = medgemma_service.health_check()
    
    return HealthResponse(
        status="healthy",
        ai_service=ai_health,
        audio_service={
            "initialized": True,
            "supported_formats": [fmt.value for fmt in AudioFormat],
            "supported_languages": list(audio_service.get_supported_languages().keys())
        },
        available_report_types=[rt.value for rt in ReportType]
    )


@router.get("/report_types")
async def get_report_types() -> Dict:
    """Get available report types with descriptions"""
    report_type_descriptions = {
        ReportType.SOAP_NOTE.value: "SOAP Note (Subjective, Objective, Assessment, Plan)",
        ReportType.PROGRESS_NOTE.value: "Progress Note (Daily clinical progress)",
        ReportType.DISCHARGE_SUMMARY.value: "Discharge Summary (Hospital discharge documentation)",
        ReportType.CONSULTATION_NOTE.value: "Consultation Note (Specialist consultation)",
        ReportType.HISTORY_PHYSICAL.value: "History & Physical (Comprehensive patient evaluation)",
        ReportType.EMERGENCY_NOTE.value: "Emergency Department Note (Emergency care documentation)",
        ReportType.OPERATIVE_NOTE.value: "Operative Note (Surgical procedure documentation)",
        ReportType.CLINICAL_REPORT.value: "Clinical Report (General medical report)"
    }
    
    return {
        "available_types": report_type_descriptions,
        "default_type": ReportType.CLINICAL_REPORT.value
    }


@router.get("/supported_audio_formats")
async def get_supported_audio_formats() -> Dict:
    """Get supported audio formats"""
    return {
        "formats": [fmt.value for fmt in AudioFormat],
        "max_size_mb": 10,
        "recommended_format": AudioFormat.WAV.value
    }


@router.get("/supported_languages")
async def get_supported_languages() -> Dict:
    """Get supported languages for speech recognition"""
    return audio_service.get_supported_languages()


@router.post("/transcribe_audio")
async def transcribe_audio_only(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form("en-US", description="Language for speech recognition")
) -> Dict:
    """
    Transcribe audio to text only (without generating report)
    
    Returns:
        Transcription result
    """
    try:
        # Read audio file
        audio_data = await audio_file.read()
        
        # Get format from filename
        audio_format = audio_file.filename.split('.')[-1].lower()
        
        # Transcribe audio
        transcribed_text, confidence = audio_service.audio_to_text(
            audio_data,
            audio_format=audio_format,
            language=language
        )
        
        if not transcribed_text:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio"
            )
        
        return {
            "transcription": transcribed_text,
            "confidence": confidence,
            "language": language,
            "audio_format": audio_format,
            "length_chars": len(transcribed_text)
        }
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))