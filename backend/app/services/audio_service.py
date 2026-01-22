import speech_recognition as sr
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple
import logging
from pydub import AudioSegment
import io

logger = logging.getLogger(__name__)


class AudioProcessingService:
    """Service for converting audio recordings to text"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        
    def convert_audio_format(self, audio_data: bytes, input_format: str = 'wav') -> bytes:
        """
        Convert audio to WAV format for speech recognition
        
        Args:
            audio_data: Raw audio bytes
            input_format: Format of input audio
            
        Returns:
            Audio in WAV format
        """
        try:
            # Create audio segment from bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)
            
            # Convert to WAV
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format='wav')
            
            return wav_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            # Return original if conversion fails
            return audio_data
    
    def audio_to_text(
        self, 
        audio_data: bytes, 
        audio_format: str = 'wav',
        language: str = 'en-US'
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Convert audio recording to text using speech recognition
        
        Args:
            audio_data: Audio file bytes
            audio_format: Format of audio file
            language: Language code for transcription
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        temp_file = None
        
        try:
            # Convert to WAV if needed
            if audio_format.lower() != 'wav':
                audio_data = self.convert_audio_format(audio_data, audio_format)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_file = temp_audio.name
            
            # Use SpeechRecognition to transcribe
            with sr.AudioFile(temp_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record the audio
                audio = self.recognizer.record(source)
                
                # Try multiple recognition engines
                try:
                    # Try Google Speech Recognition (requires internet)
                    text = self.recognizer.recognize_google(audio, language=language)
                    confidence = 0.8  # Google doesn't return confidence
                    
                except sr.UnknownValueError:
                    logger.warning("Speech recognition could not understand audio")
                    return None, None
                    
                except sr.RequestError as e:
                    logger.error(f"Could not request results from speech recognition service: {e}")
                    # Fall back to Sphinx (offline)
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                        confidence = 0.6  # Sphinx confidence
                    except:
                        logger.error("All speech recognition methods failed")
                        return None, None
            
            logger.info(f"Successfully transcribed {len(audio_data)} bytes of audio")
            return text, confidence
            
        except Exception as e:
            logger.error(f"Error in audio transcription: {e}")
            return None, None
            
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def validate_audio(self, audio_data: bytes, max_size_mb: int = 10) -> Tuple[bool, Optional[str]]:
        """
        Validate audio file
        
        Args:
            audio_data: Audio file bytes
            max_size_mb: Maximum allowed size in MB
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check size
        size_mb = len(audio_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"Audio file too large ({size_mb:.1f}MB > {max_size_mb}MB)"
        
        # Check if we can process it
        try:
            # Try to create AudioSegment to validate format
            AudioSegment.from_file(io.BytesIO(audio_data))
            return True, None
        except Exception as e:
            return False, f"Invalid audio format: {str(e)}"
    
    def get_supported_languages(self) -> dict:
        """Get supported languages for speech recognition"""
        return {
            'en-US': 'English (US)',
            'en-GB': 'English (UK)',
            'es-ES': 'Spanish',
            'fr-FR': 'French',
            'de-DE': 'German',
            'it-IT': 'Italian',
            'pt-BR': 'Portuguese (Brazil)',
            'ru-RU': 'Russian',
            'zh-CN': 'Chinese (Mandarin)',
            'ja-JP': 'Japanese',
            'ar-SA': 'Arabic',
            'hi-IN': 'Hindi'
        }


# Singleton instance
audio_service = AudioProcessingService()


def get_audio_service() -> AudioProcessingService:
    """Get the singleton audio service instance"""
    return audio_service