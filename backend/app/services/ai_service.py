import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, Dict, Any, List
import logging
import base64
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedGemmaService:
    """Service for interacting with Med-Gemma model for medical report generation"""
    
    def __init__(
        self,
        model_name: str = "google/medgemma-2b",
        use_gpu: bool = True,
        load_in_4bit: bool = True,
        max_length: int = 2048,
        temperature: float = 0.3,  # Lower temperature for medical accuracy
        top_p: float = 0.95
    ):
        """
        Initialize the Med-Gemma service
        
        Args:
            model_name: Hugging Face model name (medgemma-2b, medgemma-7b, or medgemma-27b)
            use_gpu: Whether to use GPU if available
            load_in_4bit: Use 4-bit quantization to reduce memory (recommended for 7b/27b)
            max_length: Maximum context length
            temperature: Sampling temperature (lower = more focused)
            top_p: Nucleus sampling parameter
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        self.device = None
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        self.is_initialized = False
        self.use_mock = False  # Fallback to mock if model fails to load
    
    def initialize(self):
        """Initialize the Med-Gemma model and tokenizer"""
        try:
            logger.info(f"Initializing Med-Gemma service with model: {self.model_name}")
            
            # Check if CUDA is available
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for inference")
                # For CPU, don't use 4-bit quantization
                self.load_in_4bit = False
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            logger.info("Loading Med-Gemma model...")
            
            # Configure model loading based on available resources
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
            }
            
            if self.load_in_4bit and self.device.type == "cuda":
                model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                })
                logger.info("Using 4-bit quantization for memory efficiency")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using 4-bit quantization
            if not self.load_in_4bit:
                self.model.to(self.device)
            
            self.model.eval()
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )
            
            self.is_initialized = True
            logger.info(f"Med-Gemma service initialized successfully with {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Med-Gemma model: {e}")
            logger.warning("Falling back to mock mode for development/demo")
            self.use_mock = True
            self.is_initialized = True
    
    def generate_medical_report(
        self,
        clinical_notes: str,
        report_type: str = "clinical_report",
        patient_context: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a medical report using Med-Gemma
        
        Args:
            clinical_notes: Transcribed clinical notes from doctor-patient conversation
            report_type: Type of medical report to generate
            patient_context: Optional patient information (age, gender, etc.)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary containing generated report and metadata
        """
        if not self.is_initialized:
            self.initialize()
        
        # Use instance parameters if not specified
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        
        # If using mock mode or model failed to load
        if self.use_mock or self.generator is None:
            return self._generate_mock_report(clinical_notes, report_type, patient_context)
        
        try:
            logger.info(f"Generating {report_type} report with {max_new_tokens} max tokens")
            
            # Format the prompt based on report type
            prompt = self._format_prompt(clinical_notes, report_type, patient_context)
            
            # Generate text with Med-Gemma
            with torch.no_grad():
                outputs = self.generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Prevent repetition
                )
            
            generated_text = outputs[0]['generated_text']
            
            # Extract only the newly generated part (remove the prompt)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Clean up the generated text
            generated_text = self._clean_generated_text(generated_text)
            
            # Extract sections if possible
            sections = self._extract_report_sections(generated_text, report_type)
            
            return {
                "report": generated_text,
                "sections": sections,
                "model": self.model_name,
                "report_type": report_type,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                },
                "success": True,
                "is_mock": False
            }
            
        except Exception as e:
            logger.error(f"Error generating medical report: {e}")
            # Fallback to mock generation
            return self._generate_mock_report(clinical_notes, report_type, patient_context)
    
    def _format_prompt(
        self, 
        clinical_notes: str, 
        report_type: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format prompt for Med-Gemma based on report type"""
        
        # Base instruction for Med-Gemma
        base_prompt = """You are Med-Gemma, a specialized medical AI assistant. Generate a professional medical report based on the following clinical conversation.

CLINICAL CONVERSATION TRANSCRIPTION:
{clinical_notes}

PATIENT INFORMATION:
{patient_info}

INSTRUCTIONS:
{instructions}

Generate a {report_type} following standard medical documentation guidelines:"""

        # Patient context
        patient_info = "No specific patient information provided."
        if patient_context:
            info_parts = []
            if patient_context.get("age"):
                info_parts.append(f"Age: {patient_context['age']}")
            if patient_context.get("gender"):
                info_parts.append(f"Gender: {patient_context['gender']}")
            if info_parts:
                patient_info = "; ".join(info_parts)
        
        # Report type specific instructions
        instructions_map = {
            "soap_note": """Create a SOAP note with these sections:
1. SUBJECTIVE: Patient's symptoms, history, concerns
2. OBJECTIVE: Examination findings, vital signs, test results
3. ASSESSMENT: Diagnosis, differential diagnosis, clinical reasoning
4. PLAN: Treatment plan, medications, follow-up, referrals

Format with clear headings and bullet points.""",
            
            "progress_note": """Create a Progress Note with these sections:
- Date/Time of Visit
- Chief Complaint
- Subjective: Patient's reported progress
- Objective: Current examination findings
- Assessment: Clinical assessment
- Plan: Next steps in treatment

Use professional medical terminology.""",
            
            "discharge_summary": """Create a Discharge Summary with these sections:
- Admission Date
- Discharge Date
- Reason for Admission
- Hospital Course
- Discharge Diagnoses
- Discharge Medications
- Discharge Instructions
- Follow-up Appointments

Include all relevant clinical details.""",
            
            "consultation_note": """Create a Consultation Note with these sections:
- Reason for Consultation
- History of Present Illness
- Review of Systems
- Physical Examination
- Assessment and Recommendations
- Plan

Be comprehensive yet concise.""",
            
            "history_physical": """Create a History and Physical Examination report:
- Chief Complaint
- History of Present Illness
- Past Medical History
- Medications and Allergies
- Family History
- Social History
- Review of Systems
- Physical Examination (by system)
- Assessment
- Plan

Use standard H&P format.""",
            
            "emergency_note": """Create an Emergency Department Note:
- Chief Complaint
- History of Present Illness
- Triage Information
- Physical Examination
- Diagnostic Tests
- Emergency Interventions
- Disposition
- Instructions

Focus on acute care aspects.""",
            
            "operative_note": """Create an Operative Note:
- Preoperative Diagnosis
- Postoperative Diagnosis
- Procedure Performed
- Surgeon and Assistants
- Anesthesia
- Findings
- Description of Procedure
- Estimated Blood Loss
- Complications
- Disposition

Include surgical details.""",
            
            "clinical_report": """Create a comprehensive Clinical Report:
- Patient Presentation
- Clinical Findings
- Diagnostic Results
- Clinical Assessment
- Treatment Recommendations
- Follow-up Plan
- Patient Education

Organize information clearly and professionally."""
        }
        
        instructions = instructions_map.get(
            report_type, 
            instructions_map["clinical_report"]
        )
        
        # Format the complete prompt
        prompt = base_prompt.format(
            clinical_notes=clinical_notes,
            patient_info=patient_info,
            instructions=instructions,
            report_type=report_type.replace("_", " ").title()
        )
        
        return prompt
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text"""
        # Remove any remaining prompt fragments
        text = text.replace("Generate a", "").replace("following standard", "")
        
        # Remove any hallucinated conversation markers
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith(("User:", "Assistant:", "Doctor:", "Patient:")):
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _extract_report_sections(self, report: str, report_type: str) -> Dict[str, str]:
        """Extract sections from generated report"""
        sections = {}
        
        # Section headers based on report type
        section_patterns = {
            "soap_note": ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"],
            "progress_note": ["CHIEF COMPLAINT", "SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"],
            "discharge_summary": ["ADMISSION", "DISCHARGE", "REASON", "HOSPITAL COURSE", "DIAGNOSES", "MEDICATIONS", "INSTRUCTIONS"],
            "clinical_report": ["PRESENTATION", "FINDINGS", "ASSESSMENT", "RECOMMENDATIONS", "PLAN"]
        }
        
        patterns = section_patterns.get(report_type, [])
        
        lines = report.split('\n')
        current_section = "Header"
        current_content = []
        
        for line in lines:
            line_upper = line.strip().upper()
            
            # Check if this line starts a new section
            is_section = any(
                pattern in line_upper and len(line.strip()) < 100
                for pattern in patterns
            )
            
            if is_section and line.strip().endswith(':'):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.strip().rstrip(':')
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _generate_mock_report(
        self, 
        clinical_notes: str, 
        report_type: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a mock medical report for testing/demo"""
        
        # Patient info
        patient_info = ""
        if patient_context:
            if patient_context.get("age"):
                patient_info += f"Age: {patient_context['age']} years\n"
            if patient_context.get("gender"):
                patient_info += f"Gender: {patient_context['gender']}\n"
        
        # Generate mock report based on type
        mock_reports = {
            "soap_note": f"""SOAP NOTE - MOCK GENERATION

SUBJECTIVE:
Patient reports: {clinical_notes[:100]}...
Chief Complaint: As per clinical conversation.
History of Present Illness: Based on transcribed conversation.

OBJECTIVE:
Vital Signs: Within normal limits.
Physical Examination: Unremarkable based on available information.
Diagnostic Tests: Pending review.

ASSESSMENT:
1. Provisional diagnosis requiring further evaluation.
2. Differential diagnosis to be considered.

PLAN:
1. Order appropriate diagnostic tests.
2. Initiate symptomatic treatment if indicated.
3. Schedule follow-up appointment.

Note: This is a mock report. Verify all information with actual clinical data.""",
            
            "clinical_report": f"""CLINICAL REPORT - MOCK VERSION
Generated from audio transcription

PATIENT INFORMATION:
{patient_info}

CLINICAL SUMMARY:
Based on the doctor-patient conversation: {clinical_notes[:150]}...

KEY FINDINGS:
- Symptoms reported as per conversation
- Further evaluation needed

ASSESSMENT:
Requires comprehensive clinical evaluation.

RECOMMENDATIONS:
1. Complete diagnostic workup
2. Consider specialist referral if needed
3. Patient education regarding condition

NOTE: This report is generated for demonstration. All clinical decisions must be verified.""",
        }
        
        report = mock_reports.get(
            report_type, 
            f"Medical Report ({report_type.replace('_', ' ').title()})\n\n{clinical_notes[:500]}...\n\n[This is a mock report for demonstration]"
        )
        
        return {
            "report": report,
            "sections": {"Main": report},
            "model": "medgemma-mock-1.0",
            "report_type": report_type,
            "parameters": {},
            "success": True,
            "is_mock": True
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of the AI service"""
        return {
            "initialized": self.is_initialized,
            "model_loaded": not self.use_mock,
            "model_name": self.model_name,
            "device": str(self.device) if self.device else None,
            "use_mock": self.use_mock,
            "gpu_available": torch.cuda.is_available(),
            "model_size": "2B" if "2b" in self.model_name.lower() else 
                          "7B" if "7b" in self.model_name.lower() else 
                          "27B" if "27b" in self.model_name.lower() else "Unknown"
        }


# Singleton instance
medgemma_service = MedGemmaService(model_name="google/medgemma-2b")  # Start with 2B for demo


def get_medgemma_service() -> MedGemmaService:
    """Get the singleton Med-Gemma service instance"""
    return medgemma_service