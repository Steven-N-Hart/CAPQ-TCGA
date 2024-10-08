from google.cloud import aiplatform
import logging
import argparse
logger = logging.getLogger()


logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process DICOM files and upload images to Google Cloud Storage.')
    parser.add_argument('--gcp_project_id', type=str, help='Google Cloud Project ID', default='correlation-aware-pq')
    parser.add_argument('--bq_results_csv', type=str, help='Path to the CSV file containing BQ results.', default='gs://capq-tcga/workspace/bq_results_df.csv')
    parser.add_argument('--bucket_name', type=str, help='Google Cloud Storage bucket name', default='capq-tcga')
    parser.add_argument('--dirname', type=str, help='Directory name for storing images.', default='data')
    parser.add_argument('--no-subset', dest='subset', action='store_false', help='Extract all foreground patches if set.')
    parser.add_argument('--location', type=str, help='Google Cloud Storage bucket location', default='us-central1')
    parser.add_argument('--display_name', type=str, help='Display name for Vertex AI job', default='dicom-processing-job')
    parser.add_argument('--container_uri', type=str, help='Display name for VertexAI job', default='gcr.io/correlation-aware-pq/dcm_to_img_frames_job:latest')
    parser.add_argument('--staging_bucket', type=str, help='Staging bucket', default='staging')
    parser.add_argument('--machine_type', type=str, help='Machine type', default='n1-highmem-32')
    parser.add_argument('--replica_count', type=int, help='Number of replicas', default=12)
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    # Initialize Vertex AI
    staging_bucket = f'gs://{args.bucket_name}/{args.staging_bucket}'
    logger.debug(f'Initializing Vertex AI with staging bucket: {staging_bucket}')

    aiplatform.init(project=args.gcp_project_id, location=args.location, staging_bucket=staging_bucket)
    script_path = 'dcm_to_img_frames.py'
    logger.debug(f'Script path: {script_path}')
    # Define the job arguments, including the subset flag
    job_args = [
        '--bq_results_csv', args.bq_results_csv,
        '--bucket_name', args.bucket_name,
        '--dirname', args.dirname,
    ]

    if not args.subset:
        job_args.append('--no-subset')

    # Define and run the custom job
    job = aiplatform.CustomJob.from_local_script(
        display_name=args.display_name,
        script_path=script_path,
        container_uri=args.container_uri,
        requirements=['pandas', 'Pillow', 'google-cloud-storage', 'pydicom'],
        args=job_args,
        project=args.gcp_project_id,
        location=args.location,
        machine_type=args.machine_type,
        replica_count=args.replica_count,
        labels={"display_name": args.display_name},
        )

    logger.debug(f'Submitting job...')
    job.run()
