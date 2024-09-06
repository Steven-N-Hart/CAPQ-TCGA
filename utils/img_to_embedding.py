import argparse
from google.cloud import storage, bigquery
from google.api_core.retry import Retry
from google.api_core.exceptions import Conflict, NotFound  # Import the Conflict and NotFound exceptions

import torch
from PIL import Image
import io
import logging
import os
from dotenv import load_dotenv

load_dotenv()
from dpfm_model_runners.model_factory import model_factory  # Can't load until HuggingFace token is exposed

logger = logging.getLogger()

retry = Retry(
    initial=1.0,
    maximum=10.0,
    multiplier=2.0,
    deadline=120.0,
)


def create_bigquery_table_if_not_exists(table_name, bq_client):
    try:
        # Check if the table exists
        bq_client.get_table(table_name)
        logger.info(f"Table {table_name} already exists.")
    except NotFound:
        # Table does not exist, create it
        schema = [
            bigquery.SchemaField("image_name", "STRING"),
            bigquery.SchemaField("SeriesInstanceUID", "STRING"),
            bigquery.SchemaField("SOPInstanceUID", "STRING"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        ]
        table = bigquery.Table(table_name, schema=schema)
        try:
            bq_client.create_table(table)
            logger.info(f"Table {table_name} created.")
        except Conflict:
            # Handle the case where the table was created by another process
            logger.info(f"Table {table_name} was already created by another process.")
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            raise e

def upload_to_bigquery(rows, dataset_name, table_name, bq_client, project_id):
    table_id = f'{project_id}.{dataset_name}.{table_name}'

    # Ensure the table exists before inserting data
    create_bigquery_table_if_not_exists(table_id, bq_client)

    try:
        errors = bq_client.insert_rows_json(table_id, rows, retry=retry)
    except Exception as e:
        logger.error(f'{e}')
        logger.error(f'table_id: {table_id}')
        logger.error(f'rows: {rows}')
        raise e

    if errors:
        logger.error(f"Encountered errors while inserting rows: {errors}")


def main(project_id, bucket_name, folder_prefix, dataset_name, table_name, model_name, batch_size):
    # Initialize the Google Cloud clients
    storage_client = storage.Client()
    bq_client = bigquery.Client(project=project_id)

    # Initialize the Hugging Face model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:{device}")

    device = torch.device("cuda")
    model, processor, get_image_embedding = model_factory(model_name=model_name)
    bucket = storage_client.bucket(bucket_name)

    # Find only those that match the regular expression
    blobs = [b for b in bucket.list_blobs() if b.name.startswith(folder_prefix) and b.name.endswith('.png')]

    # Process each image in the bucket
    rows_to_insert = []
    i = 0

    for blob in blobs:
        image_data = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_data))
        print(f"Device going into model:{device}")
        embedding = get_image_embedding(image, processor, model, device)

        # Prepare row for BigQuery
        row = {
            "image_name": '/'.join(blob.name.split('/')[1:]),
            'SeriesInstanceUID': os.path.dirname(blob.name).replace(folder_prefix, '').replace('/', ''),
            'SOPInstanceUID': os.path.basename(blob.name).replace(folder_prefix, '').replace('/', '').split('_')[0],
            "embedding": embedding.tolist()
        }
        rows_to_insert.append(row)
        i += 1

        if i % batch_size == 0:
            upload_to_bigquery(rows_to_insert, dataset_name, table_name, bq_client, project_id)
            rows_to_insert.clear()  # Clear the batch
            logger.info(f"Uploaded {i} rows")

    # Upload any remaining rows
    if rows_to_insert:
        upload_to_bigquery(rows_to_insert, dataset_name, table_name, bq_client, project_id)
        logger.info(f"Uploaded remaining {len(rows_to_insert)} rows")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Submit AI training Job to process images and store embeddings in BigQuery.')
    parser.add_argument('--bucket_name', type=str, help='Google Cloud Storage bucket name', default='capq-tcga')
    parser.add_argument('--folder_prefix', type=str, help='Folder prefix in the bucket to look for images',
                        default='images')
    parser.add_argument('--dataset_name', type=str, help='BigQuery Dataset Name', default='tcga')
    parser.add_argument('--table_name', type=str, help='BigQuery Table Name', default='phikon')
    parser.add_argument('--model_name', type=str, help='Hugging Face Model Name', default='owkin/phikon',
                        choices=['owkin/phikon', 'paige-ai/Virchow2', 'MahmoodLab/conch',
                                 'prov-gigapath/prov-gigapath', 'LGAI-EXAONE/EXAONEPath','histai/hibou-L','histai/hibou-b'])
    parser.add_argument('--project_id', type=str, help='Project ID', default='correlation-aware-pq')
    parser.add_argument('--batch_size', type=int, help='Batch size to insert into BQ', default=500)
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    main(args.project_id, args.bucket_name, args.folder_prefix, args.dataset_name, args.table_name, args.model_name, args.batch_size)
