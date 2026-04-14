"""
enhancer.py - AI-powered speech enhancement

Supports TWO AI engines:
1. DeepFilterNet  — Fast, lightweight CPU denoiser (8 MB model)
2. Resemble Enhance — State-of-the-art denoiser + super-resolution
   (restores lost frequencies, makes it sound like a better mic)

Users can select the engine from the UI dropdown.
"""

import os
import tempfile
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global model caches
_dfn_model = None
_dfn_state = None

# Enhancement strength presets for DeepFilterNet (atten_lim_db values)
ENHANCEMENT_PRESETS = {
    "gentle": 10,
    "balanced": 15,
    "aggressive": 25,
    "maximum": None,
}

# Available AI engines
AI_ENGINES = ["DeepFilterNet", "Resemble Enhance"]


def _load_deepfilternet():
    """Load DeepFilterNet model (cached)."""
    global _dfn_model, _dfn_state
    if _dfn_model is not None and _dfn_state is not None:
        return _dfn_model, _dfn_state

    try:
        from df import init_df
        logger.info("Loading DeepFilterNet model...")
        model, df_state, _ = init_df()
        _dfn_model = model
        _dfn_state = df_state
        logger.info("DeepFilterNet model loaded successfully.")
        return model, df_state
    except Exception as e:
        logger.error(f"Failed to load DeepFilterNet: {e}")
        raise RuntimeError(f"Could not load DeepFilterNet: {e}") from e


def enhance_with_ai(input_wav_path: str, strength: str = "balanced",
                    engine: str = "DeepFilterNet") -> str:
    """
    Enhance a WAV audio file using the selected AI engine.

    Args:
        input_wav_path: Path to a WAV file.
        strength: Enhancement strength preset (for DeepFilterNet).
        engine: AI engine to use ("DeepFilterNet" or "Resemble Enhance").

    Returns:
        Path to the AI-enhanced WAV file.
    """
    if engine == "Resemble Enhance":
        return _enhance_with_resemble(input_wav_path)
    else:
        return _enhance_with_deepfilternet(input_wav_path, strength)


def _enhance_with_deepfilternet(input_wav_path: str,
                                 strength: str = "balanced") -> str:
    """Enhance using DeepFilterNet with attenuation limiting."""
    try:
        import torch
        import torchaudio
        from df import enhance

        model, df_state = _load_deepfilternet()

        atten_lim = ENHANCEMENT_PRESETS.get(strength, 15)
        if atten_lim is not None:
            logger.info(f"DeepFilterNet: strength='{strength}', atten_lim={atten_lim} dB")
        else:
            logger.info(f"DeepFilterNet: strength='{strength}', no attenuation limit")

        audio, sr = torchaudio.load(input_wav_path)

        expected_sr = df_state.sr()
        if sr != expected_sr:
            logger.info(f"Resampling from {sr} Hz to {expected_sr} Hz")
            resampler = torchaudio.transforms.Resample(sr, expected_sr)
            audio = resampler(audio)

        duration_sec = audio.shape[-1] / expected_sr
        logger.info(f"Processing {duration_sec:.1f}s of audio...")

        if duration_sec > 600:
            enhanced_audio = _process_dfn_chunks(
                model, df_state, audio, expected_sr, atten_lim
            )
        else:
            logger.info("Running DeepFilterNet enhancement...")
            enhanced_audio = enhance(model, df_state, audio, atten_lim_db=atten_lim)

        temp_file = tempfile.NamedTemporaryFile(
            suffix="_ai_enhanced.wav", delete=False, dir=tempfile.gettempdir()
        )
        output_path = temp_file.name
        temp_file.close()

        torchaudio.save(output_path, enhanced_audio, expected_sr)
        logger.info(f"DeepFilterNet enhancement complete: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"DeepFilterNet enhancement failed: {e}")
        raise RuntimeError(f"DeepFilterNet enhancement failed: {e}") from e


def _enhance_with_resemble(input_wav_path: str) -> str:
    """
    Enhance using Resemble Enhance — state-of-the-art model.
    Does BOTH denoising AND super-resolution (bandwidth extension).
    """
    try:
        import torch
        import torchaudio
        from resemble_enhance.enhancer.inference import denoise, enhance as resemble_enhance

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Resemble Enhance: loading audio (device={device})...")

        # Load audio
        wav, sr = torchaudio.load(input_wav_path)

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(0)
        else:
            wav = wav.squeeze(0)

        duration_sec = wav.shape[-1] / sr
        logger.info(f"Resemble Enhance: processing {duration_sec:.1f}s of audio...")

        # Step 1: Denoise first (fast, removes background noise)
        logger.info("Resemble Enhance: Step 1/2 - Denoising...")
        wav_denoised, new_sr = denoise(wav, sr, device=device)

        # Step 2: Enhance (super-resolution, restores frequencies)
        logger.info("Resemble Enhance: Step 2/2 - Super-resolution enhancement...")
        # nfe=32 is a good balance between quality and speed on CPU
        # Higher values (64, 128) = better quality but much slower
        nfe = 32 if device == "cpu" else 64
        wav_enhanced, new_sr = resemble_enhance(
            wav_denoised, new_sr, device=device, nfe=nfe, solver="midpoint"
        )

        # Save output
        temp_file = tempfile.NamedTemporaryFile(
            suffix="_resemble_enhanced.wav", delete=False, dir=tempfile.gettempdir()
        )
        output_path = temp_file.name
        temp_file.close()

        # Ensure correct shape for saving [C, T]
        if wav_enhanced.dim() == 1:
            wav_enhanced = wav_enhanced.unsqueeze(0)

        torchaudio.save(output_path, wav_enhanced.cpu(), new_sr)
        logger.info(f"Resemble Enhance complete: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Resemble Enhance failed: {e}")
        raise RuntimeError(f"Resemble Enhance failed: {e}") from e


def _process_dfn_chunks(model, df_state, audio, sr, atten_lim_db,
                        chunk_duration_sec=300, overlap_sec=5):
    """Process long audio in overlapping chunks with crossfade."""
    import torch
    from df import enhance

    total_samples = audio.shape[-1]
    chunk_samples = int(chunk_duration_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step = chunk_samples - overlap_samples

    logger.info(f"Chunked processing: {chunk_duration_sec}s chunks, {overlap_sec}s overlap")

    chunks = []
    start = 0
    chunk_idx = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio[:, start:end]
        chunk_idx += 1
        logger.info(f"  Chunk {chunk_idx}: {start/sr:.1f}s - {end/sr:.1f}s")

        if hasattr(model, "reset_h0"):
            model.reset_h0(batch_size=chunk.shape[0], device=chunk.device)

        enhanced_chunk = enhance(model, df_state, chunk, atten_lim_db=atten_lim_db)
        chunks.append((start, end, enhanced_chunk))
        start += step

    # Stitch with crossfade
    output = torch.zeros_like(audio)
    weights = torch.zeros(1, total_samples)

    for start, end, chunk in chunks:
        chunk_len = chunk.shape[-1]
        actual_end = start + chunk_len
        w = torch.ones(1, chunk_len)

        if start > 0:
            fade_len = min(overlap_samples, chunk_len)
            w[:, :fade_len] = torch.linspace(0.0, 1.0, fade_len)
        if actual_end < total_samples:
            fade_len = min(overlap_samples, chunk_len)
            w[:, -fade_len:] = torch.linspace(1.0, 0.0, fade_len)

        output[:, start:actual_end] += chunk[:, :actual_end-start] * w[:, :actual_end-start]
        weights[:, start:actual_end] += w[:, :actual_end-start]

    weights = weights.clamp(min=1e-8)
    output = output / weights
    logger.info(f"Chunked processing complete: {chunk_idx} chunks merged.")
    return output


def is_model_available() -> bool:
    """Check if DeepFilterNet is available."""
    try:
        from df import init_df  # noqa: F401
        return True
    except ImportError:
        return False


def is_resemble_available() -> bool:
    """Check if Resemble Enhance is available."""
    try:
        from resemble_enhance.enhancer.inference import denoise  # noqa: F401
        return True
    except ImportError:
        return False
