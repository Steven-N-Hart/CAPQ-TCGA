# Deploying Containers for use with VertexAI training jobs

In the `../utils` folder are a series of python files that end in `*.job.py`. These are specifically designed to run
VertexAI training jobs.  However, some will need additional packages that aren't available in the standard Docker 
containers offered by Google.


## `dcm_to_img_frames_job.py`
```shell
cd ~/PycharmProjects/CAPQ-TCGA
gcloud auth configure-docker
docker build -t gcr.io/correlation-aware-pq/dcm_to_img_frames_job -f docker/dcm_to_img_frames_job.Dockerfile .
docker push gcr.io/correlation-aware-pq/dcm_to_img_frames_job
```