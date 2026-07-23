import glob
import os
import sys

# Automatically discover and expose all NVIDIA package bin directories to Windows PATH
site_pkgs = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
for bin_path in glob.glob(os.path.join(site_pkgs, "nvidia", "*", "bin")):
  if os.path.exists(bin_path) and bin_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + bin_path

# Prevent OpenMP multi-threading collisions on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import torch
from dotenv import load_dotenv
import whisperx

# Load .env file for HF_TOKEN
load_dotenv()

# --- Configuration ---
AUDIO_DIR = Path("audio")
MODEL_NAME = "medium"  # Model size: tiny, base, small, medium, large-v3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
  raise ValueError(
      "HF_TOKEN is missing! Check your .env file or environment variables."
  )

AUDIO_EXTENSIONS = {".amr", ".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def process_audio_files():
  if not AUDIO_DIR.exists():
    print(f"Error: Directory '{AUDIO_DIR}' does not exist.")
    return

  audio_files = [
      f for f in AUDIO_DIR.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS
  ]

  if not audio_files:
    print(f"No audio files found recursively in '{AUDIO_DIR}'.")
    return

  print(
      f"Found {len(audio_files)} audio file(s). Initializing WhisperX on"
      f" {DEVICE.upper()}..."
  )

  try:
    print(f"Loading Whisper model ({MODEL_NAME})...")
    model = whisperx.load_model(
        MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE
    )

    print("Loading speaker diarization pipeline...")
    diarize_model = whisperx.diarize.DiarizationPipeline(
        token=HF_TOKEN, device=DEVICE
    )
  except Exception as e:
    print(f"Initialization Error: {e}")
    return

  for audio_path in audio_files:
    print(f"\n--- Processing: {audio_path.name} ---")
    try:
      if DEVICE == "cuda":
        torch.cuda.empty_cache()

      print(" [1/4] Loading audio into memory...")
      audio = whisperx.load_audio(str(audio_path))

      print(" [2/4] Transcribing audio...")
      result = model.transcribe(audio, batch_size=8)

      print(" [3/4] Aligning timestamps...")
      model_a, metadata = whisperx.load_align_model(
          language_code=result["language"], device=DEVICE
      )
      result = whisperx.align(
          result["segments"],
          model_a,
          metadata,
          audio,
          DEVICE,
          return_char_alignments=False,
      )

      print(" [4/4] Running speaker diarization...")
      diarize_segments = diarize_model(audio)
      result = whisperx.assign_word_speakers(diarize_segments, result)

      output_path = audio_path.with_suffix(".json")
      with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

      print(f" Successfully saved to: {output_path}")

      print("\nSpeaker Transcript Summary:")
      for segment in result.get("segments", []):
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        print(f" [{speaker}]: {text}")

    except Exception as e:
      print(f" Error processing {audio_path.name}: {e}")


if __name__ == "__main__":
  process_audio_files()