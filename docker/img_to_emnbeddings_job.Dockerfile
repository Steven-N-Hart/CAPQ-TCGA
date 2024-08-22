FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest
ENV PATH="$PATH:/root/.local/bin"

COPY utils /

RUN pip install dpfm_factory

ENTRYPOINT ["python", "/utils/img_to_embedding.py"]