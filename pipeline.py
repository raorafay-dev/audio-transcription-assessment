import whisper
from pydub import AudioSegment
import os
import math

class TranscriptionPipeline:
    def __init__(self, model_size="base"):
        print(f"Loading Whisper '{model_size}' model...")
        # Engineering Decision: Load model once on initialization
        self.model = whisper.load_model(model_size)
        self.chunk_length_ms = 5 * 60 * 1000  # 5 minutes per chunk to prevent OOM

    def normalize_audio(self, input_path: str, output_path: str) -> str:
        """Converts any audio format to 16kHz mono WAV."""
        print(f"Normalizing audio format for: {input_path}")
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")
        return output_path

    def process_audio(self, file_path: str):
        """Handles chunking, transcription, and timestamp aggregation."""
        temp_wav = "temp_normalized.wav"
        
        try:
            # 1. Normalize the format
            self.normalize_audio(file_path, temp_wav)
            audio = AudioSegment.from_file(temp_wav)
            
            total_length_ms = len(audio)
            final_segments = []
            full_text = ""
            
            # 2. Chunking for long files
            num_chunks = math.ceil(total_length_ms / self.chunk_length_ms)
            print(f"Processing {num_chunks} chunk(s)...")

            for i in range(num_chunks):
                start_ms = i * self.chunk_length_ms
                end_ms = min((i + 1) * self.chunk_length_ms, total_length_ms)
                
                chunk = audio[start_ms:end_ms]
                chunk_path = f"temp_chunk_{i}.wav"
                chunk.export(chunk_path, format="wav")
                
                # 3. Transcribe chunk
                result = self.model.transcribe(chunk_path)
                full_text += result["text"] + " "
                
                # 4. Adjust timestamps
                time_offset = start_ms / 1000.0  # Convert ms to seconds
                for segment in result["segments"]:
                    final_segments.append({
                        "start": round(segment["start"] + time_offset, 2),
                        "end": round(segment["end"] + time_offset, 2),
                        "text": segment["text"].strip()
                    })
                
                # Cleanup chunk
                os.remove(chunk_path)

            return {
                "status": "success",
                "full_text": full_text.strip(),
                "segments": final_segments
            }

        except Exception as e:
            print(f"Error processing audio: {e}")
            return {"status": "error", "message": str(e)}
            
        finally:
            # Always clean up the normalized temp file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
