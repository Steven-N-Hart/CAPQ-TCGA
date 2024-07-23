# noinspection PyPackageRequirements
from google.cloud import storage
import logging

logger = logging.getLogger()


def download_public_file(bucket_name, source_blob_name, destination_file_name, local=True):
    """Downloads a public blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    if local:
        blob.download_to_filename(destination_file_name)
    else:
        return blob

    logging.info(
        "Downloaded public blob {} from bucket {} to {}.".format(
            source_blob_name, bucket.name, destination_file_name
        )
    )


def copy_blob_using_rewrite(source_bucket_name, source_blob_name, destination_bucket_name, destination_blob_name):
    """
    Copy a blob from one bucket to another using the rewrite method if it doesn't already exist.

    :param source_bucket_name: Name of the source bucket.
    :param source_blob_name: Name of the source blob.
    :param destination_bucket_name: Name of the destination bucket.
    :param destination_blob_name: Name of the destination blob.
    """
    storage_client = storage.Client()

    source_bucket = storage_client.bucket(source_bucket_name)
    source_blob = source_bucket.blob(source_blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)
    destination_blob = destination_bucket.blob(destination_blob_name)

    # Check if the destination blob already exists
    if destination_blob.exists():
        logger.info('Blob %s already exists in bucket %s. Skipping copy.', destination_blob_name, destination_bucket_name)
        return

    # Use the rewrite method to copy the blob
    token = None
    while True:
        token, _, _ = destination_blob.rewrite(source_blob, token=token)
        if token is None:
            break
    logger.info('Completed copy of %s to %s', source_blob_name, destination_blob_name)


def write_array_to_bucket(bucket_name):
    # Instantiates a client
    storage_client = storage.Client()
    bucket = client.bucket(bucket_name)