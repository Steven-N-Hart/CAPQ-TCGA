FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest

COPY utils /

RUN pip install pandas Pillow google-cloud-storage

ENTRYPOINT ["python", "/utils/dcm_to_img_job.py"]