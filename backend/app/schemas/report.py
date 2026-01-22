from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union
from datetime import datetime
from enum import Enum


class ReportType(str, Enum):
    """Enum for supported report types"""
    SOAP_NOTE = "soap_note"
    PROGRESS_NOTE = "progress_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    CONSULTATION_NOTE = "consultation_note"
    HISTORY_PHYSICAL = "history_physical"
    EMERGENCY_NOTE = "emergency_note"
    OPERATIVE_NOTE = "operative_note"
    CLINICAL_REPORT = "clinical_report"


class AudioFormat(str, Enum):
    """Enum for supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"


class ReportRequest(BaseModel):
    """Request model for generating a report from clinical notes"""
    clinical_notes: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Clinical notes from the healthcare provider"
    )
    
    report_type: ReportType = Field(
        ReportType.CLINICAL_REPORT,
        description="Type of medical report to generate"
    )
    
    language: str = Field(
        "en-US",
        description="Language code for the report"
    )
    
    include_plan: bool = Field(
        True,
        description="Include treatment plan in the report"
    )
    
    include_medications: bool = Field(
        True,
        description="Include medications section in the report"
    )
    
    patient_age: Optional[int] = Field(
        None,
        ge=0,
        le=120,
        description="Patient age (optional, for better context)"
    )
    
    patient_gender: Optional[str] = Field(
        None,
        description="Patient gender (optional, for better context)"
    )


class AudioReportRequest(BaseModel):
    """Request model for generating a report from audio recording"""
    audio_data: str = Field(
        ...,
        description="Base64 encoded audio file"
    )
    
    audio_format: AudioFormat = Field(
        AudioFormat.WAV,
        description="Format of the audio file"
    )
    
    report_type: ReportType = Field(
        ReportType.CLINICAL_REPORT,
        description="Type of medical report to generate"
    )
    
    language: str = Field(
        "en-US",
        description="Language code for speech recognition and report"
    )
    
    include_plan: bool = Field(
        True,
        description="Include treatment plan in the report"
    )
    
    include_medications: bool = Field(
        True,
        description="Include medications section in the report"
    )
    
    patient_age: Optional[int] = Field(
        None,
        ge=0,
        le=120,
        description="Patient age (optional, for better context)"
    )
    
    patient_gender: Optional[str] = Field(
        None,
        description="Patient gender (optional, for better context)"
    )


class ReportSection(BaseModel):
    """Individual section of a medical report"""
    title: str
    content: str
    confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Confidence score for this section"
    )


class TranscriptionResult(BaseModel):
    """Result of audio transcription"""
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    language: str = Field(..., description="Language detected")


class ReportResponse(BaseModel):
    """Response model for generated report"""
    report: str = Field(
        ...,
        description="Complete generated medical report"
    )
    
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Individual sections of the report"
    )
    
    transcription: Optional[TranscriptionResult] = Field(
        None,
        description="Audio transcription result (if audio input was used)"
    )
    
    metadata: Dict = Field(
        default_factory=dict,
        description="Metadata about the generation process"
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of report generation"
    )
    
    model_used: str = Field(
        ...,
        description="Name of the AI model used for generation"
    )
    
    report_type: str = Field(
        ...,
        description="Type of report generated"
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    ai_service: Dict = Field(..., description="AI service health")
    audio_service: Dict = Field(..., description="Audio service health")
    available_report_types: List[str] = Field(..., description="Available report types")
    timestamp: datetime = Field(default_factory=datetime.now)