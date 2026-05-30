"""
extractor.py - Audio extraction from video files

Uses FFmpeg to extract the audio track from video files
at maximum quality, preserving the original codec when possible.
"""

import os
import tempfile
import logging
import subprocess

logger = logging.getLogger(__name__)

# Supported video formats
VIDEO_FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}


def extract_audio(video_path: str, output_format: str = "wav") -> str:
    """
    Extract audio from a video file using FFmpeg.

    Args:
        video_path: Path to the video file.
        output_format: Output audio format (wav, mp3, flac).

    Returns:
        Path to the extracted audio file.
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError("No se proporcionó un archivo de video válido.")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in VIDEO_FORMATS:
        raise ValueError(
            f"Formato de video no soportado: {ext}. "
            f"Válidos: {', '.join(sorted(VIDEO_FORMATS))}"
        )

    # Create output file
    temp_file = tempfile.NamedTemporaryFile(
        suffix=f"_extracted.{output_format}", delete=False,
        dir=tempfile.gettempdir()
    )
    output_path = temp_file.name
    temp_file.close()

    logger.info(f"Extracting audio from: {os.path.basename(video_path)}")

    # Build FFmpeg command for max quality extraction
    if output_format == "wav":
        # PCM 24-bit for maximum quality
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                    # No video
            "-acodec", "pcm_s24le",   # 24-bit PCM
            "-ar", "48000",           # 48kHz sample rate
            output_path
        ]
    elif output_format == "flac":
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",
            "-acodec", "flac",
            "-compression_level", "8",
            "-ar", "48000",
            output_path
        ]
    else:  # mp3
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "0",              # Best quality VBR
            "-ar", "48000",
            output_path
        ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr[-500:]}")

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(
            f"Audio extracted successfully: {output_path} "
            f"({file_size_mb:.1f} MB)"
        )
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out (>5 min).")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and add it to PATH."
        )


def get_video_info(video_path: str) -> dict:
    """Get basic info about a video file using FFprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            streams = data.get("streams", [])
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
            return {
                "duration": float(fmt.get("duration", 0)),
                "size_mb": int(fmt.get("size", 0)) / (1024 * 1024),
                "format": fmt.get("format_long_name", "Unknown"),
                "has_audio": has_audio,
            }
    except Exception:
        pass
    return {"duration": 0, "size_mb": 0, "format": "Unknown", "has_audio": True}
