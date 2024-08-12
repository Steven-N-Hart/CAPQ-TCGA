import argparse
from google.cloud import storage, bigquery
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import io
import logging
import os
logger = logging.getLogger()



def get_image_embedding(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()

def upload_to_bigquery(rows, dataset_name, table_name, bq_client):
    table_id = f'{dataset_name}.{table_name}'
    try:
        errors = bq_client.insert_rows_json(table_id, rows)
        if errors:
            logger.error(f"Encountered errors while inserting rows: {errors}")
    except Exception as e:
        logger.error(e)

def main(project_id, bucket_name, folder_prefix, dataset_name, table_name, model_name):
    # Initialize the Google Cloud clients
    storage_client = storage.Client()
    bq_client = bigquery.Client(project=project_id)

    # Initialize the Hugging Face model and processor
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)

    bucket = storage_client.bucket(bucket_name)

    # List all blobs in the bucket
    blobs = bucket.list_blobs(prefix=folder_prefix)

    # Process each image in the bucket
    rows_to_insert = []
    for blob in blobs:
        if blob.name.endswith('.png'):
            image_data = blob.download_as_bytes()
            image = Image.open(io.BytesIO(image_data))
            embedding = get_image_embedding(image, processor, model)

            # Prepare row for BigQuery
            row = {
                "image_name": '/'.join(blob.name.split('/')[1:]),
                'SeriesInstanceUID': os.path.dirname(blob.name).replace(folder_prefix, '').replace('/',''),
                'SOPInstanceUID': os.path.basename(blob.name).replace(folder_prefix, '').replace('/',''),
                "embedding": embedding.tolist()
            }
            rows_to_insert.append(row)

    # Upload embeddings to BigQuery
    upload_to_bigquery(rows_to_insert, dataset_name, table_name, bq_client)
    logger.info("Embeddings uploaded successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit AI training Job to process images and store embeddings in BigQuery.')
    parser.add_argument('--bucket_name', type=str, help='Google Cloud Storage bucket name', default='capq-tcga')
    parser.add_argument('--folder_prefix', type=str, help='Folder prefix in the bucket to look for images', default='workspace')
    parser.add_argument('--dataset_name', type=str, help='BigQuery Dataset Name', default='tcga')
    parser.add_argument('--table_name', type=str, help='BigQuery Table Name', default='phikon')
    parser.add_argument('--model_name', type=str, help='BigQuery Dataset Name', default='owkin/phikon')
    parser.add_argument('--project_id', type=str, help='Project ID', default='correlation-aware-pq')
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    main(args.project_id, args.bucket_name, args.folder_prefix, args.dataset_name, args.table_name, args.model_name)
