FROM python:3.11-slim

# System dependencies
#  - ffmpeg / libsndfile1-dev : audio I/O
#  - espeak-ng                : phonemizer required by Kokoro TTS (incl. Spanish)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1-dev \
    espeak-ng \
    build-essential \
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

# --- Voice cloning (CPU): OpenVoice V2 + MeloTTS ---
# Installed last so pip resolves their version-sensitive deps after the rest.
# This is the most conflict-prone layer; pin versions here while iterating on
# the HF build if the resolver complains.
RUN pip install --no-cache-dir --user \
        git+https://github.com/myshell-ai/MeloTTS.git \
        git+https://github.com/myshell-ai/OpenVoice.git \
        faster-whisper \
    && python -m unidic download || echo "MeloTTS/OpenVoice install needs version pinning (iterate on build)."

# Copy app code (owned by the runtime user) and create writable output dir
COPY --chown=user . .
RUN mkdir -p outputs

# Pre-download models so the first user request is fast (all non-fatal:
# they re-download at runtime into the user-owned HF cache if skipped).
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='e'); KPipeline(lang_code='a')" || \
    echo "Kokoro pre-warm skipped (will download on first use)."
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('myshell-ai/OpenVoiceV2')" || \
    echo "OpenVoice V2 checkpoints will download on first use."

# HF Spaces expects port 7860
EXPOSE 7860

ENV PYTHONIOENCODING=utf-8 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
# Optional premium engine: set OPENAI_API_KEY as a Space secret to enable
# "OpenAI Studio" (gpt-4o-mini-tts). Without it, only Edge + Kokoro show up.
# Optional cost guard: OPENAI_DAILY_CHAR_BUDGET (default 50000 chars/day).

CMD ["python", "app.py"]
