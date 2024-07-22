import argparse
from google.cloud import storage, bigquery
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import io
import logging
from dotenv import load_dotenv
load_dotenv("../.env")

logger = logging.getLogger()
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
assert GCP_PROJECT_ID is not None, 'GCP_PROJECT_ID must be set'


def get_image_embedding(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()

def upload_to_bigquery(rows, dataset_name, table_name, bq_client):
    table_id = f'{dataset_name}.{table_name}'
    errors = bq_client.insert_rows_json(table_id, rows)
    if errors:
        logger.error(f"Encountered errors while inserting rows: {errors}")

def main(bucket_name, folder_prefix, dataset_name, table_name, model_name):
    # Initialize the Google Cloud clients
    storage_client = storage.Client()
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)

    # Initialize the Hugging Face model and processor
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, use_fast=True)

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
                "image_name": blob.name,
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

    args = parser.parse_args()
    main(args.bucket_name, args.folder_prefix, args.dataset_name, args.table_name, args.model_name)
