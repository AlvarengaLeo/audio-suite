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

# Copy app code (owned by the runtime user) and create writable output dir
COPY --chown=user . .
RUN mkdir -p outputs

# Pre-download the Kokoro model so the first user request is fast.
# Non-fatal: if the download fails at build time it retries at runtime
# (the cache is user-owned, so runtime downloads also succeed).
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='e'); KPipeline(lang_code='a')" || \
    echo "Kokoro pre-warm skipped (will download on first use)."

# HF Spaces expects port 7860
EXPOSE 7860

ENV PYTHONIOENCODING=utf-8 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
# Optional premium engine: set OPENAI_API_KEY as a Space secret to enable
# "OpenAI Studio" (gpt-4o-mini-tts). Without it, only Edge + Kokoro show up.
# Optional cost guard: OPENAI_DAILY_CHAR_BUDGET (default 50000 chars/day).

CMD ["python", "app.py"]
