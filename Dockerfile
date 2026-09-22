FROM python:3.12-slim

# Create user to run the app (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user

# Set home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Install CPU PyTorch and dependencies
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -e .

EXPOSE 7860

# Run uvicorn on port 7860
CMD ["uvicorn", "cropguard.inference.service:app", "--host", "0.0.0.0", "--port", "7860"]
