# Audio Transcription Pipeline

A robust Python pipeline for transcribing audio files, extracting segment-level timestamps, and handling edge cases like varied audio formats and long-duration files.

## Engineering Decisions & Architecture

1. **Format Normalization (`pydub` / FFmpeg):** 
   Audio formats in production are highly variable (OGG, M4A, variable bitrates). Feeding these directly into an ML model causes unpredictable behavior. This pipeline intercepts all uploads and normalizes them to a strict `16kHz mono WAV` format before inference, ensuring deterministic model behavior.

2. **Memory Management via Chunking:** 
   Processing long audio files (e.g., 1+ hours) synchronously causes Out-Of-Memory (OOM) crashes. This pipeline splits normalized audio into 5-minute chunks, processes them sequentially, and mathematically offsets the timestamps during aggregation to ensure perfect alignment with the master file.

3. **Model State Management:** 
   The Whisper model is loaded once during class initialization (`__init__`) rather than per-transcription. In a production API environment, this prevents the heavy compute cost of reloading the model on every request.

## Setup & Installation

Ensure you have `ffmpeg` installed on your system, as `pydub` relies on it for audio manipulation.

```bash
# Install system dependencies (Ubuntu/Debian example)
sudo apt update && sudo apt install ffmpeg

# Install Python dependencies
pip install -r requirements.txt
