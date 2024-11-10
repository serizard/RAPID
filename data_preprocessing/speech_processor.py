import whisper_timestamped as whisper
import json
import numpy as np
from typing import Dict, List, Optional
import torch
import jiwer
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os
import time
from datetime import datetime

class AphasiaSpeechProcessor:
    def __init__(self, model_size: str = "turbo", device: str = "cuda"):
        """
        Initialize the speech processor with specified model size and device.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large", "turbo")
            device: Computing device ("cuda" or "cpu")
        """
        self.device = device
        self.model = whisper.load_model(model_size, device=self.device)
        self.model_size = model_size

    def process_with_retry(
        self,
        func,
        *args,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        error_log_file: str = "transcribing_errors.txt",
        **kwargs
    ):
        """
        Generic retry mechanism for processing functions.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            error_log_file: Path to error log file
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = (
                    f"Error in {func.__name__} "
                    f"(Attempt {attempt + 1}/{max_retries}): {str(e)}\n"
                    f"Args: {args}, Kwargs: {kwargs}\n"
                    f"Timestamp: {datetime.now().isoformat()}\n"
                )
                with open(error_log_file, "a") as log_file:
                    log_file.write(error_message)
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
    
    def batch_process_audio(
        self,
        audio_dir: str,
        output_dir: str,
        error_log_file: str = "error_log.txt",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        continue_on_error: bool = True
    ) -> List[Dict]:
        """
        Process multiple audio files with retry mechanism.
        
        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save transcription results
            error_log_file: Path to error log file
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            continue_on_error: Whether to continue processing other files if one fails
            
        Returns:
            List of processing results for each audio file
        """
        audio_paths = glob(os.path.join(audio_dir, "*.mp3"))
        results = []
        
        # Initialize error log with session information
        with open(error_log_file, "w") as log_file:
            log_file.write(
                f"Batch Processing Session Started: {datetime.now().isoformat()}\n"
                f"Model: {self.model_size}, Max Retries: {max_retries}, "
                f"Retry Delay: {retry_delay}s\n\n"
            )
        
        for audio_path in tqdm(audio_paths, desc='Processing audio files'):
            try:
                # Process the audio file with retry
                result = self.process_with_retry(
                    process_aphasia_recording,
                    audio_path,
                    model_size=self.model_size,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    error_log_file=error_log_file
                )
                
                if result:
                    # Add processing attempts to metadata
                    result["metadata"]["processing_attempts"] = 1  # Will be >1 if retries were needed
                    
                    # Save individual result
                    output_file = os.path.join(
                        output_dir,
                        f"{os.path.splitext(os.path.basename(audio_path))[0]}_result.json"
                    )
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    results.append(result)
                elif not continue_on_error:
                    raise Exception(f"Failed to process {audio_path} after {max_retries} attempts")
                
            except Exception as e:
                error_message = (
                    f"Fatal error processing {audio_path}: {str(e)}\n"
                    f"Timestamp: {datetime.now().isoformat()}\n"
                )
                with open(error_log_file, "a") as log_file:
                    log_file.write(error_message)
                
                if not continue_on_error:
                    raise
        
        # Write processing summary
        with open(error_log_file, "a") as log_file:
            log_file.write(
                f"\nBatch Processing Session Completed: {datetime.now().isoformat()}\n"
                f"Total files processed: {len(results)}/{len(audio_paths)}\n"
            )
        
        return results

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio array
        """
        try:
            audio = whisper.load_audio(audio_path, sr=16000)
            return audio
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")

    def transcribe_audio(
        self, 
        audio: np.ndarray,
        language: str = "en",
        detect_disfluencies: bool = True
    ) -> Dict:
        """
        Transcribe audio with timestamps and disfluency detection.
        
        Args:
            audio: Audio array to transcribe
            language: Language code
            detect_disfluencies: Whether to detect speech disfluencies
            
        Returns:
            Dictionary containing transcription results
        """
        return whisper.transcribe(
            self.model,
            audio,
            detect_disfluencies=detect_disfluencies,
            language=language
        )

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate between reference and hypothesis transcriptions.
        
        Args:
            reference: Reference transcription
            hypothesis: Hypothesis transcription
            
        Returns:
            Word Error Rate score
        """
        return jiwer.wer(reference, hypothesis)

    def chunk_transcription(
        self,
        result: Dict,
        chunk_size: int = 50,
        min_duration: float = 3.0
    ) -> List[Dict]:
        """
        Chunk transcription into segments based on token count and duration.
        
        Args:
            result: Transcription result dictionary
            chunk_size: Number of tokens per chunk
            min_duration: Minimum chunk duration in seconds
            
        Returns:
            List of chunked transcriptions
        """
        chunks = []
        current_chunk = {
            "text": "",
            "tokens": [],
            "start": None,
            "end": None
        }
        
        token_count = 0
        
        for segment in result["segments"]:
            for word in segment["words"]:
                if token_count == 0:
                    current_chunk["start"] = word["start"]
                
                current_chunk["tokens"].append(word)
                current_chunk["text"] += f" {word['text']}"
                token_count += 1
                
                if token_count >= chunk_size:
                    current_chunk["end"] = word["end"]
                    duration = current_chunk["end"] - current_chunk["start"]
                    
                    if duration >= min_duration:
                        chunks.append(current_chunk)
                    
                    current_chunk = {
                        "text": "",
                        "tokens": [],
                        "start": None,
                        "end": None
                    }
                    token_count = 0
        
        if token_count > 0:
            current_chunk["end"] = word["end"]
            duration = current_chunk["end"] - current_chunk["start"]
            if duration >= min_duration:
                chunks.append(current_chunk)
                
        return chunks

def process_aphasia_recording(
    audio_path: str,
    reference_text: Optional[str] = None,
    model_size: str = "medium",
    chunk_size: int = 50,
    min_duration: float = 3.0,
    output_dir: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Dict:
    """
    Process an aphasia recording with complete pipeline and retry mechanism.
    
    Args:
        audio_path: Path to audio file
        reference_text: Optional reference transcription for WER calculation
        model_size: Whisper model size
        chunk_size: Number of tokens per chunk
        min_duration: Minimum chunk duration in seconds
        output_dir: Optional directory to save transcription results
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Dictionary containing processing results
    """
    processor = AphasiaSpeechProcessor(model_size=model_size)

    # Process audio with retry mechanism
    audio = processor.process_with_retry(
        processor.preprocess_audio,
        audio_path,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    if audio is None:
        raise Exception(f"Failed to preprocess audio from {audio_path}")
    
    result = processor.process_with_retry(
        processor.transcribe_audio,
        audio,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    if result is None:
        raise Exception(f"Failed to transcribe audio from {audio_path}")
    
    # Calculate WER if reference is provided
    wer_score = None
    if reference_text:
        hypothesis = " ".join(segment["text"] for segment in result["segments"])
        wer_score = processor.process_with_retry(
            processor.calculate_wer,
            reference_text,
            hypothesis,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    # Chunk the transcription
    chunks = processor.process_with_retry(
        processor.chunk_transcription,
        result,
        chunk_size=chunk_size,
        min_duration=min_duration,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    # Prepare final results
    final_result = {
        "transcription": result,
        "chunks": chunks,
        "wer": wer_score,
        "metadata": {
            "audio_path": audio_path,
            "model_size": model_size,
            "chunk_size": chunk_size,
            "min_duration": min_duration,
            "processing_timestamp": datetime.now().isoformat()
        }
    }
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(audio_path))[0]}_result.json"
        )
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save results to {output_file}: {str(e)}")
    
    return final_result

# Example usage
if __name__ == "__main__":
    audio_dir = "/workspace/dataset/audio_clips"
    output_dir = "/workspace/dataset/transcripts_re"
    
    processor = AphasiaSpeechProcessor(model_size="turbo")
    results = processor.batch_process_audio(
        audio_dir=audio_dir,
        output_dir=output_dir,
        max_retries=3,
        retry_delay=1.0,
        continue_on_error=True
    )
    
    # Save combined results
    try:
        with open(os.path.join(output_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save combined results: {str(e)}")