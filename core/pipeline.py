"""
pipeline.py - Main audio enhancement pipeline

Orchestrates the full enhancement flow:
1. Validate input → 2. Convert format → 3. AI denoise → 4. Podcast postprocess → 5. Export

Supports TWO AI engines:
- DeepFilterNet: Fast, lightweight, CPU-optimized
- Resemble Enhance: State-of-the-art, denoising + super-resolution

Handles errors gracefully: if one engine fails, it falls back
to the other, then to FFmpeg-only processing.
"""

import os
import shutil
import logging
import tempfile
from datetime import datetime

from core.converter import convert_to_working_format, get_audio_info
from core.enhancer import (enhance_with_ai, is_model_available,
                           is_resemble_available, ENHANCEMENT_PRESETS, AI_ENGINES)
from core.postprocessor import apply_podcast_postprocessing, POSTPROCESS_PRESETS

logger = logging.getLogger(__name__)

# Output directory
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def ensure_outputs_dir():
    """Create the outputs directory if it doesn't exist."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _generate_output_filename(original_path: str) -> str:
    """Generate a clean output filename based on the original file."""
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    clean_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in base_name)
    clean_name = clean_name.strip().replace(" ", "_")
    if not clean_name:
        clean_name = "audio"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{clean_name}_enhanced_{timestamp}.wav"


def run_pipeline(input_path: str,
                 enhancement_strength: str = "balanced",
                 postprocess_preset: str = "natural",
                 ai_engine: str = "DeepFilterNet",
                 progress_callback=None) -> tuple[str, str]:
    """
    Run the full audio enhancement pipeline.

    Args:
        input_path: Path to the user's original audio file.
        enhancement_strength: AI enhancement strength
            ("gentle", "balanced", "aggressive", "maximum").
        postprocess_preset: Postprocessing preset
            ("natural", "podcast", "broadcast").
        ai_engine: AI engine ("DeepFilterNet" or "Resemble Enhance").
        progress_callback: Optional callable(float, str) for progress updates.

    Returns:
        Tuple of (output_file_path, status_message).
    """
    if progress_callback is None:
        progress_callback = lambda p, m: None  # noqa: E731

    ensure_outputs_dir()

    # --- Step 1: Validate input ---
    progress_callback(0.05, "Validando archivo de entrada...")

    if not input_path or not os.path.isfile(input_path):
        raise ValueError("No se proporcionó un archivo de audio válido. Sube un archivo de audio.")

    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    if file_size_mb > 500:
        raise ValueError(
            f"El archivo es demasiado grande ({file_size_mb:.1f} MB). El tamaño máximo es 500 MB."
        )

    input_info = get_audio_info(input_path)
    logger.info(f"Input file info: {input_info}")

    # --- Step 2: Convert to working format ---
    progress_callback(0.15, "Convirtiendo a formato de trabajo (48 kHz mono WAV)...")
    converted_path = convert_to_working_format(input_path)
    logger.info(f"Converted to working format: {converted_path}")

    # --- Step 3: AI Enhancement ---
    ai_used = False
    engine_used = "FFmpeg"
    enhanced_path = converted_path

    # Try selected engine first
    if ai_engine == "Resemble Enhance" and is_resemble_available():
        try:
            progress_callback(
                0.30,
                "Ejecutando Resemble Enhance (denoising + super-resolucion)..."
            )
            enhanced_path = enhance_with_ai(
                converted_path, strength=enhancement_strength,
                engine="Resemble Enhance"
            )
            ai_used = True
            engine_used = "Resemble Enhance"
            logger.info("Resemble Enhance succeeded.")
        except Exception as e:
            logger.warning(f"Resemble Enhance failed: {e}")
            progress_callback(0.30, "Resemble fallo. Intentando DeepFilterNet...")

    # Fallback to DeepFilterNet if needed
    if not ai_used and is_model_available():
        try:
            atten = ENHANCEMENT_PRESETS.get(enhancement_strength, 15)
            atten_label = f"{atten} dB" if atten else "sin limite"
            progress_callback(
                0.30,
                f"Ejecutando DeepFilterNet ({enhancement_strength}: {atten_label}) — "
                f"puede tardar varios minutos en CPU..."
            )
            enhanced_path = enhance_with_ai(
                converted_path, strength=enhancement_strength,
                engine="DeepFilterNet"
            )
            ai_used = True
            engine_used = "DeepFilterNet"
            logger.info("DeepFilterNet enhancement succeeded.")
        except Exception as e:
            logger.warning(f"DeepFilterNet failed: {e}")
            progress_callback(0.30, "IA fallo. Usando FFmpeg como respaldo...")

    if not ai_used:
        logger.warning("No AI engine available. Using FFmpeg-only pipeline.")
        progress_callback(0.30, "IA no disponible. Usando FFmpeg...")

    # --- Step 4: Postprocessing ---
    progress_callback(
        0.65,
        f"Aplicando masterizacion (preset: {postprocess_preset})..."
    )
    postprocessed_path = apply_podcast_postprocessing(
        enhanced_path, preset=postprocess_preset
    )
    logger.info(f"Postprocessing complete: {postprocessed_path}")

    # --- Step 5: Export ---
    progress_callback(0.90, "Guardando audio mejorado...")
    output_filename = _generate_output_filename(input_path)
    final_output_path = os.path.join(OUTPUTS_DIR, output_filename)
    shutil.copy2(postprocessed_path, final_output_path)

    _cleanup_temp_files(converted_path, enhanced_path, postprocessed_path)
    progress_callback(1.0, "Mejora completada!")

    status = f"Mejorado con {engine_used}."
    logger.info(status)
    return final_output_path, status


def _cleanup_temp_files(*paths):
    """Remove temporary files created during processing."""
    for path in paths:
        try:
            if path and os.path.exists(path) and tempfile.gettempdir() in path:
                os.remove(path)
        except OSError:
            pass
