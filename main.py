from pipeline import TranscriptionPipeline
import json
import sys

def main():
    # Check if user provided a file argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_audio_file>")
        return

    file_path = sys.argv[1]
    
    # Initialize the pipeline
    pipeline = TranscriptionPipeline(model_size="base")
    
    # Process the audio
    print(f"Processing file: {file_path}")
    result = pipeline.process_audio(file_path)
    
    # Output the result beautifully
    print("\n--- TRANSCRIPTION RESULT ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
