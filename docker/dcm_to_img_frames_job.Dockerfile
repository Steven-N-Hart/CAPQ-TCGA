FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest
ENV PATH="$PATH:/root/.local/bin"

COPY utils /

RUN pip install pandas Pillow google-cloud-storage

ENTRYPOINT ["python", "/utils/dcm_to_img_job.py"]