"""
postprocessor.py - Podcast-style audio postprocessing

Uses Spotify's 'pedalboard' library for professional audio effects.
Pedalboard is chosen because:
- Created by Spotify's Audio Intelligence Lab
- Provides broadcast-quality audio processing
- Supports VST-style effects: EQ, compression, limiting, gain
- Very fast, written in C++ with Python bindings
- Produces natural-sounding results without metallic artifacts

The postprocessing chain simulates a typical podcast mastering chain:
1. High-pass filter  → removes low-frequency rumble
2. Parametric EQ     → subtle vocal shaping
3. Compressor        → gentle dynamic control
4. Gain              → volume adjustment
5. Limiter           → prevents clipping

PHILOSOPHY: Less is more. We want to enhance clarity without
making the voice sound "processed". These settings are intentionally
conservative — a natural-sounding result is always better than
an over-processed one.
"""

import os
import tempfile
import logging
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Postprocessing intensity presets
POSTPROCESS_PRESETS = {
    "natural": {
        # Minimal processing — just cleanup and normalization
        "highpass_hz": 60.0,
        "low_shelf_gain": -1.0,
        "high_shelf_gain": 1.5,
        "comp_threshold": -24.0,
        "comp_ratio": 2.0,
        "comp_attack": 15.0,
        "comp_release": 200.0,
        "makeup_gain": 2.0,
        "limiter_threshold": -1.0,
    },
    "podcast": {
        # Standard podcast mastering — balanced and professional
        "highpass_hz": 75.0,
        "low_shelf_gain": -1.5,
        "high_shelf_gain": 2.0,
        "comp_threshold": -20.0,
        "comp_ratio": 2.5,
        "comp_attack": 12.0,
        "comp_release": 180.0,
        "makeup_gain": 3.0,
        "limiter_threshold": -1.0,
    },
    "broadcast": {
        # More processing — radio/broadcast style
        "highpass_hz": 80.0,
        "low_shelf_gain": -2.0,
        "high_shelf_gain": 2.5,
        "comp_threshold": -18.0,
        "comp_ratio": 3.0,
        "comp_attack": 10.0,
        "comp_release": 150.0,
        "makeup_gain": 4.0,
        "limiter_threshold": -1.0,
    },
}


def apply_podcast_postprocessing(input_wav_path: str,
                                  preset: str = "natural") -> str:
    """
    Apply podcast-style mastering chain to a WAV file.

    This mimics what a professional podcast editor would do:
    clean up lows, add vocal clarity, compress dynamics, and
    normalize loudness — all while keeping the voice natural.

    Presets:
    - "natural"   — Minimal processing, maximum voice preservation
    - "podcast"   — Standard podcast mastering, balanced
    - "broadcast" — Heavier processing, radio-style

    Args:
        input_wav_path: Path to a WAV file (ideally already AI-enhanced).
        preset: Postprocessing preset name.

    Returns:
        Path to the postprocessed WAV file.
    """
    try:
        from pedalboard import (
            Pedalboard,
            HighpassFilter,
            Compressor,
            Gain,
            Limiter,
            LowShelfFilter,
            HighShelfFilter,
        )
        from pedalboard.io import AudioFile

        # Get preset settings
        settings = POSTPROCESS_PRESETS.get(preset, POSTPROCESS_PRESETS["natural"])
        logger.info(
            f"Applying '{preset}' postprocessing preset with pedalboard..."
        )

        # Read input audio
        with AudioFile(input_wav_path) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate

        # Build the mastering chain with preset values
        board = Pedalboard([
            # 1. High-pass filter — removes rumble, handling noise, wind
            HighpassFilter(cutoff_frequency_hz=settings["highpass_hz"]),

            # 2. Low shelf — subtle mud reduction in low-mids
            LowShelfFilter(
                cutoff_frequency_hz=200.0,
                gain_db=settings["low_shelf_gain"]
            ),

            # 3. High shelf — gentle presence/air boost for clarity
            HighShelfFilter(
                cutoff_frequency_hz=3500.0,
                gain_db=settings["high_shelf_gain"]
            ),

            # 4. Gentle compressor — smooth dynamics, NOT squash them
            #    Key: Slow attack lets transients through naturally
            #    Key: Low ratio preserves dynamic range
            Compressor(
                threshold_db=settings["comp_threshold"],
                ratio=settings["comp_ratio"],
                attack_ms=settings["comp_attack"],
                release_ms=settings["comp_release"],
            ),

            # 5. Makeup gain — compensate for compression
            Gain(gain_db=settings["makeup_gain"]),

            # 6. Limiter — safety net, prevents clipping
            Limiter(
                threshold_db=settings["limiter_threshold"],
                release_ms=120.0,
            ),
        ])

        # Process the audio through the chain
        processed_audio = board(audio, sample_rate)

        # Save the output
        temp_file = tempfile.NamedTemporaryFile(
            suffix="_postprocessed.wav", delete=False, dir=tempfile.gettempdir()
        )
        output_path = temp_file.name
        temp_file.close()

        sf.write(output_path, processed_audio.T, sample_rate, subtype="PCM_16")
        logger.info(f"Postprocessing complete: {output_path}")
        return output_path

    except ImportError:
        logger.warning("pedalboard not available, using FFmpeg fallback")
        return _ffmpeg_fallback_postprocessing(input_wav_path, preset)
    except Exception as e:
        logger.warning(f"Pedalboard postprocessing failed: {e}. Using FFmpeg fallback.")
        return _ffmpeg_fallback_postprocessing(input_wav_path, preset)


def _ffmpeg_fallback_postprocessing(input_wav_path: str,
                                     preset: str = "natural") -> str:
    """
    Fallback postprocessing using FFmpeg filters when pedalboard is not available.

    Applies:
    - High-pass filter
    - Vocal presence EQ boost
    - Dynamic range compression
    - Loudness normalization to -16 LUFS (podcast standard)
    """
    import subprocess

    settings = POSTPROCESS_PRESETS.get(preset, POSTPROCESS_PRESETS["natural"])
    logger.info(f"Applying FFmpeg fallback postprocessing (preset: {preset})...")

    temp_file = tempfile.NamedTemporaryFile(
        suffix="_ffmpeg_processed.wav", delete=False, dir=tempfile.gettempdir()
    )
    output_path = temp_file.name
    temp_file.close()

    # FFmpeg filter chain — matched to preset values
    hp = settings["highpass_hz"]
    ls_gain = settings["low_shelf_gain"]
    hs_gain = settings["high_shelf_gain"]
    comp_thresh = settings["comp_threshold"]
    comp_ratio = settings["comp_ratio"]
    comp_attack = settings["comp_attack"]
    comp_release = settings["comp_release"]
    makeup = settings["makeup_gain"]

    filter_chain = (
        f"highpass=f={hp},"
        f"equalizer=f=200:t=h:w=200:g={ls_gain},"
        f"equalizer=f=3500:t=h:w=1000:g={hs_gain},"
        f"acompressor=threshold={comp_thresh}dB:ratio={comp_ratio}"
        f":attack={comp_attack}:release={comp_release},"
        f"volume={makeup}dB,"
        "loudnorm=I=-16:TP=-1:LRA=11"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav_path,
        "-af", filter_chain,
        "-ar", "48000",
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        logger.info(f"FFmpeg fallback postprocessing complete: {output_path}")
        return output_path
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg is not installed or not in PATH. "
            "Please install FFmpeg: https://ffmpeg.org/download.html"
        )
    except Exception as e:
        raise RuntimeError(f"FFmpeg postprocessing failed: {e}") from e
