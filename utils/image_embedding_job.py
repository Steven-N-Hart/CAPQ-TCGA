from google.cloud import aiplatform
import logging
import argparse
from dotenv import load_dotenv
load_dotenv()
import os

logger = logging.getLogger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and store embeddings in BigQuery.')
    parser.add_argument('--gcp_project_id', type=str, help='Google Cloud Project ID', default='correlation-aware-pq')
    parser.add_argument('--bucket_name', type=str, help='Google Cloud Storage bucket name', default='capq-tcga')
    parser.add_argument('--folder_prefix', type=str, help='Folder prefix in the bucket to look for images', default='images')
    parser.add_argument('--dataset_name', type=str, help='BigQuery Dataset Name', default='tcga')
    parser.add_argument('--table_name', type=str, help='BigQuery Table Name', default='phikon')
    parser.add_argument('--model_name', type=str, help='BigQuery Dataset Name', default='owkin/phikon')
    parser.add_argument('--location', type=str, help='Google Cloud Storage bucket location', default='us-central1')
    parser.add_argument('--display_name', type=str, help='Display name for VertexAI job', default='image-embedding-job')
    parser.add_argument('--container_uri', type=str, help='Display name for VertexAI job', default='gcr.io/correlation-aware-pq/img_to_embeddings_job:latest')
    parser.add_argument('--staging_bucket', type=str, help='Staging bucket', default='staging')
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='DEBUG')
    parser.add_argument('--machine_type', type=str, help='Machine type', default='n1-highmem-32')
    parser.add_argument('--replica_count', type=int, help='Number of replicas', default=12)
    parser.add_argument('--accelerator', help='Number of replicas', default='NVIDIA_TESLA_P100')
    parser.add_argument('--accelerator_count', type=int, help='Number of GPUs', default=4)

    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    # Initialize Vertex AI
    staging_bucket = f'gs://{args.bucket_name}/{args.staging_bucket}'
    # Initialize Vertex AI
    logger.debug(f'Initializing Vertex AI with staging bucket: {staging_bucket}')

    aiplatform.init(project=args.gcp_project_id, location=args.location, staging_bucket=staging_bucket)
    script_path = 'utils/img_to_embedding.py'
    logger.debug(f'Script path: {script_path}')

    # Define and run the custom job
    job = aiplatform.CustomJob.from_local_script(
        display_name=args.display_name,
        environment_variables={"HUGGINGFACE_TOKEN": os.environ['HUGGINGFACE_TOKEN']},
        script_path=script_path,
        container_uri=args.container_uri,
        requirements=['transformers', 'google-cloud-storage', 'google-cloud-bigquery', 'google-cloud-resource-manager', 'google-api-core', 'accelerate','python-dotenv', 'huggingface_hub', 'dpfm_factory', 'conch_fork'],
        args=[
            '--bucket_name', args.bucket_name,
            '--folder_prefix', args.folder_prefix,
            '--dataset_name', args.dataset_name,
            '--table_name', args.table_name,
            '--model_name', args.model_name,
        ],
        project=args.gcp_project_id,
        location=args.location,
        machine_type=args.machine_type,
        replica_count=args.replica_count,
        accelerator_count=args.accelerator_count,
        accelerator_type=args.accelerator
    )
    logger.debug(f'Submitting job...')
    job.run()
