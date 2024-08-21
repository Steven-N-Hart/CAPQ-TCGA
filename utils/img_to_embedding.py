import argparse
from google.cloud import storage, bigquery
from model_runners.model_factory import model_factory
import utils
import torch
from PIL import Image
import io
import logging
import os
logger = logging.getLogger()





def upload_to_bigquery(rows, dataset_name, table_name, bq_client):
    table_id = f'{dataset_name}.{table_name}'
    try:
        errors = bq_client.insert_rows_json(table_id, rows, retry=utils.retry)
    except Exception as e:
        logger.error(f'{e}')
        logger.error(f'table_id: {table_id}')
        logger.error(f'rows: {rows}')
        raise e

    if errors:
        logger.error(f"Encountered errors while inserting rows: {errors}")

def main(project_id, bucket_name, folder_prefix, dataset_name, table_name, model_name):
    # Initialize the Google Cloud clients
    storage_client = storage.Client()
    bq_client = bigquery.Client(project=project_id)

    # Initialize the Hugging Face model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, get_image_embedding = model_factory(model_name=model_name)
    bucket = storage_client.bucket(bucket_name)

    # Find only those that match the regular expression
    blobs = [b for b in bucket.list_blobs() if b.name.startswith(folder_prefix) and b.name.endswith('.png')]

    # Process each image in the bucket
    rows_to_insert = []
    i = 0
    batch_size = 500  # Set your batch size

    for blob in blobs:
        image_data = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_data))
        embedding = get_image_embedding(image, processor, model, device)

        # Prepare row for BigQuery
        row = {
            "image_name": '/'.join(blob.name.split('/')[1:]),
            'SeriesInstanceUID': os.path.dirname(blob.name).replace(folder_prefix, '').replace('/',''),
            'SOPInstanceUID': os.path.basename(blob.name).replace(folder_prefix, '').replace('/','').split('_')[0],
            "embedding": embedding.tolist()
        }
        rows_to_insert.append(row)
        i += 1

        if i % batch_size == 0:
            upload_to_bigquery(rows_to_insert, dataset_name, table_name, bq_client)
            rows_to_insert.clear()  # Clear the batch
            logger.info(f"Uploaded {i} rows")

    # Upload any remaining rows
    if rows_to_insert:
        upload_to_bigquery(rows_to_insert, dataset_name, table_name, bq_client)
        logger.info(f"Uploaded remaining {len(rows_to_insert)} rows")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit AI training Job to process images and store embeddings in BigQuery.')
    parser.add_argument('--bucket_name', type=str, help='Google Cloud Storage bucket name', default='capq-tcga')
    parser.add_argument('--folder_prefix', type=str, help='Folder prefix in the bucket to look for images', default='images')
    parser.add_argument('--dataset_name', type=str, help='BigQuery Dataset Name', default='tcga')
    parser.add_argument('--table_name', type=str, help='BigQuery Table Name', default='phikon')
    parser.add_argument('--model_name', type=str, help='Hugging Face Model Name', default='owkin/phikon')
    parser.add_argument('--project_id', type=str, help='Project ID', default='correlation-aware-pq')
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    main(args.project_id, args.bucket_name, args.folder_prefix, args.dataset_name, args.table_name, args.model_name)
