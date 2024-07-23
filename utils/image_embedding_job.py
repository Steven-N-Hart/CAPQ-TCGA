from google.cloud import aiplatform
import os
import logging
import argparse
from dotenv import load_dotenv
load_dotenv("../.env")
logger = logging.getLogger()


GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
assert GCP_PROJECT_ID is not None, 'GCP_PROJECT_ID must be set'

logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and store embeddings in BigQuery.')
    parser.add_argument('--bucket_name', type=str, help='Google Cloud Storage bucket name', default='capq-tcga')
    parser.add_argument('--folder_prefix', type=str, help='Folder prefix in the bucket to look for images', default='workspace')
    parser.add_argument('--dataset_name', type=str, help='BigQuery Dataset Name', default='tcga')
    parser.add_argument('--table_name', type=str, help='BigQuery Table Name', default='phikon')
    parser.add_argument('--model_name', type=str, help='BigQuery Dataset Name', default='owkin/phikon')
    parser.add_argument('--location', type=str, help='Google Cloud Storage bucket location', default='us-central1')
    parser.add_argument('--display_name', type=str, help='Display name for VertexAI job', default='image-embedding-job')
    parser.add_argument('--container_uri', type=str, help='Display name for VertexAI job', default='us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest')
    parser.add_argument('--staging_bucket', type=str, help='Staging bucket', default='staging')

    args = parser.parse_args()
    # Initialize Vertex AI
    staging_bucket = f'gs://{args.bucket_name}/{args.staging_bucket}'
    # Initialize Vertex AI
    logger.debug(f'Initializing Vertex AI with staging bucket: {staging_bucket}')

    aiplatform.init(project=GCP_PROJECT_ID, location=args.location, staging_bucket=staging_bucket)
    script_path = 'img_to_embedding.py'
    logger.debug(f'Script path: {script_path}')

    # Define and run the custom job
    job = aiplatform.CustomJob.from_local_script(
        display_name=args.display_name,
        script_path=script_path,
        container_uri=args.container_uri,
        requirements=['transformers', 'google-cloud-storage', 'google-cloud-bigquery','google-cloud-resource-manager', 'python-dotenv'],
        args=[
            '--bucket_name', args.bucket_name,
            '--folder_prefix', args.folder_prefix,
            '--dataset_name', args.dataset_name,
            '--table_name', args.table_name,
            '--model_name', args.model_name,
        ],
        project=GCP_PROJECT_ID,
        location=args.location
    )
    logger.debug(f'Submitting job...')
    job.run()
