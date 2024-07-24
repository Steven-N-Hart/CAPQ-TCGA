import argparse
from google.cloud import bigquery, storage
import logging

logger = logging.getLogger()


def run_query_and_save_to_gcs(project_id, query_file, bucket_name, workspace_dir, output_filename):
    # Read the query from the file
    with open(query_file, 'r') as file:
        query = file.read()

    # Initialize BigQuery client
    bq_client = bigquery.Client(project=project_id)

    # Run the query
    logger.debug(f'Running query from file: {query_file}')
    query_job = bq_client.query(query)
    bq_results_df = query_job.to_dataframe()

    # Save the results to a CSV file in the GCS bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f'{workspace_dir}/{output_filename}')
    csv_data = bq_results_df.to_csv(index=False)
    blob.upload_from_string(csv_data, content_type='text/csv')
    logger.info(f'Saved query results to gs://{bucket_name}/{workspace_dir}/{output_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a BigQuery query and save results to GCS.')
    parser.add_argument('--gcp_project_id', type=str, help='Google Cloud Project ID', default='correlation-aware-pq')
    parser.add_argument('--bucket_name', type=str, help='Google Cloud Storage bucket name', default='capq-tcga')
    parser.add_argument('--query_file', type=str, help='Path to the file containing the BigQuery SQL query',
                        default='get_imgs.sql')
    parser.add_argument('--workspace_dir', type=str, help='Directory in the bucket to save the results', default='workspace')
    parser.add_argument('--output_filename', type=str, help='Name of the output CSV file', default='bq_results_df.csv')
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    run_query_and_save_to_gcs(
        project_id=args.project_id,
        query_file=args.query_file,
        bucket_name=args.bucket_name,
        workspace_dir=args.workspace_dir,
        output_filename=args.output_filename
    )
