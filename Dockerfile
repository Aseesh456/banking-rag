FROM python:3.13-slim

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default GPU build.
# Versions are pinned to the ones tested locally; unpinned installs pulled newer,
# incompatible transformers / sentence-transformers releases.
RUN pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu

# Install dependencies first so Docker caches this layer when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user (required by Hugging Face Spaces, good practice everywhere).
# The app needs to write the FAISS index and download models, so give it a home and ownership.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user

EXPOSE 8000

# The Gemini key is NOT baked into the image; pass it at run time with --env-file
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
