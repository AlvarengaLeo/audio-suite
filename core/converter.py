"""
converter.py - Audio format conversion and preprocessing

Handles converting input audio files to a uniform working format
using pydub (backed by FFmpeg). Ensures all audio entering the
pipeline is in the correct sample rate and channel configuration.
"""

import os
import tempfile
from pydub import AudioSegment

# DeepFilterNet expects 48 kHz mono audio
TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1  # mono for speech enhancement


def convert_to_working_format(input_path: str) -> str:
    """
    Convert any supported audio file to a standardized WAV format
    suitable for the AI enhancement model.

    Args:
        input_path: Path to the input audio file (wav, mp3, m4a, flac).

    Returns:
        Path to the converted WAV file (48 kHz, mono, 16-bit PCM).

    Raises:
        ValueError: If the file format is not supported.
        RuntimeError: If conversion fails.
    """
    supported_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"}
    ext = os.path.splitext(input_path)[1].lower()

    if ext not in supported_extensions:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {', '.join(sorted(supported_extensions))}"
        )

    try:
        # Load audio using pydub (requires FFmpeg for non-wav formats)
        audio = AudioSegment.from_file(input_path)

        # Convert to mono
        audio = audio.set_channels(TARGET_CHANNELS)

        # Resample to target sample rate
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

        # Set sample width to 16-bit (2 bytes)
        audio = audio.set_sample_width(2)

        # Export to a temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(
            suffix="_converted.wav", delete=False, dir=tempfile.gettempdir()
        )
        temp_path = temp_file.name
        temp_file.close()

        audio.export(temp_path, format="wav")
        return temp_path

    except Exception as e:
        raise RuntimeError(
            f"Error converting audio file '{os.path.basename(input_path)}': {e}"
        ) from e


def get_audio_info(file_path: str) -> dict:
    """
    Get basic information about an audio file.

    Returns:
        Dictionary with duration_seconds, sample_rate, channels, format.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        return {
            "duration_seconds": round(len(audio) / 1000.0, 2),
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "sample_width_bits": audio.sample_width * 8,
        }
    except Exception:
        return {"error": "Could not read audio info"}
