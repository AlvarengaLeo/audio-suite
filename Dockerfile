FROM python:3.11-slim

# System dependencies
#  - ffmpeg / libsndfile1-dev : audio I/O
#  - espeak-ng                : phonemizer required by Kokoro TTS (incl. Spanish)
#  - pkg-config + libav*-dev   : needed to build PyAV (OpenVoice -> faster-whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1-dev \
    espeak-ng \
    build-essential \
    pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as a fixed non-root UID 1000.
# Create that user up front so caches and outputs are writable at runtime.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

USER user
WORKDIR /home/user/app

# Install CPU-only PyTorch FIRST so the ~2 GB of CUDA wheels are never pulled
# (the default PyPI torch wheel ships nvidia-* deps that are useless on CPU).
RUN pip install --no-cache-dir --user \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Voice cloning (CPU): OpenVoice ToneColorConverter in an ISOLATED venv ---
# OpenVoice/MeloTTS pin ancient deps (gradio 3.48, transformers 4.27, numpy 1.22)
# that would BREAK the main Gradio-6 app if installed in-place. We isolate them
# in their own venv and call it via subprocess. We do NOT use MeloTTS — the base
# voice comes from the main app's Kokoro; OpenVoice only transfers the timbre.
# Non-fatal: if this venv fails to build, the rest of the app keeps working.
# OpenVoice pins faster-whisper==0.9.0 -> av==10 (PyAV), which does NOT build on
# the trixie base (ffmpeg 7). So we install OpenVoice with --no-deps and supply
# a MODERN, wheel-based dependency set (modern faster-whisper ships an av wheel).
RUN python -m venv /home/user/clone-venv && \
    /home/user/clone-venv/bin/pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    /home/user/clone-venv/bin/pip install --no-cache-dir \
        numpy soundfile librosa wavmark faster-whisper huggingface_hub \
        pydub whisper-timestamped inflect unidecode eng_to_ipa \
        pypinyin cn2an jieba langid && \
    /home/user/clone-venv/bin/pip install --no-cache-dir --no-deps \
        git+https://github.com/myshell-ai/OpenVoice.git && \
    echo "clone-venv ready." || echo "clone-venv build failed (cloning will be disabled)."

# Copy app code (owned by the runtime user) and create writable output dir
COPY --chown=user . .
RUN mkdir -p outputs

# Pre-download models so the first user request is fast (all non-fatal:
# they re-download at runtime into the user-owned HF cache if skipped).
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='e'); KPipeline(lang_code='a')" || \
    echo "Kokoro pre-warm skipped (will download on first use)."
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('myshell-ai/OpenVoiceV2')" || \
    echo "OpenVoice V2 checkpoints will download on first use."

# Pre-download the whisper models used by Audio → Texto so the first request is
# fast: 'base' (default S1) and 'large-v3' (S2 max-quality rectification). The
# Space has network at runtime, so these are NON-fatal — any other size (tiny/
# small/medium) downloads on demand into the HF cache.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')" || \
    echo "Whisper 'base' will download on first use."
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')" || \
    echo "Whisper 'large-v3' will download on first use."

# HF Spaces expects port 7860
EXPOSE 7860

ENV PYTHONIOENCODING=utf-8 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
# Optional premium engine: set OPENAI_API_KEY as a Space secret to enable
# "OpenAI Studio" (gpt-4o-mini-tts). Without it, only Edge + Kokoro show up.
# Optional cost guard: OPENAI_DAILY_CHAR_BUDGET (default 50000 chars/day).

CMD ["python", "app.py"]
