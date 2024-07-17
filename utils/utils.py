import pandas as pd
from typing import Any, Dict, List
from google.cloud import storage
import os
import logging

logger = logging.getLogger()


def get_data(blobs: List[str], bucket_name: str, outname: str = None) -> None:
    """
    Download data from Google Cloud Storage.

    Args:
        blobs (List[str]): List of blob names to download.
        bucket_name (str): Name of the GCP bucket.
        outname (str): If set, rename the downloaded files.

    Returns:
        None
    """
    client = storage.Client()

    for blob_name in blobs:
        local_name = os.path.basename(blob_name)
        if outname:
            local_name = outname

        if not os.path.exists(local_name):
            logger.info(f'Downloading {blob_name} to {local_name}')
            blob = client.get_bucket(bucket_name).get_blob(blob_name)
            blob.download_to_filename(local_name)

def create_slides_metadata(bq_results_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Builds a dataframe comprising all slides' metadata.

    Parameters
    ----------
    bq_results_df: pd.DataFrame
        Dataframe obtained from BigQuery. Contains one DICOM file (one level of a slide) per row.

    Returns
    -------
    pd.DataFrame
        Slides metadata table with one row per slide.
    """
    slides_metadata = dict()

    for index, row in bq_results_df.iterrows():
        slide_metadata = row.to_dict()
        image_id = slide_metadata['digital_slide_id']

        # Move level specific values through "pop()"
        level_data = {
            'width': slide_metadata.pop('width', None),
            'height': slide_metadata.pop('height', None),
            'pixel_spacing': slide_metadata.pop('pixel_spacing', None),
            'compression': slide_metadata.pop('compression', None),
            'crdc_instance_uuid': slide_metadata.pop('crdc_instance_uuid', None),
            'gcs_url': slide_metadata.pop('gcs_url', None)
        }

        if not image_id in slides_metadata:
            slides_metadata[image_id] = slide_metadata
            slides_metadata[image_id]['levels'] = []

        slides_metadata[image_id]['levels'].append(level_data)

    for slide_metadata in slides_metadata.values():
        slide_metadata['levels'].sort(key=lambda x: x['pixel_spacing'])

        if len(slide_metadata['levels']) > 0:
            base_level = slide_metadata['levels'][0]
            slide_metadata['width'] = base_level['width']
            slide_metadata['height'] = base_level['height']

    return pd.DataFrame.from_records(list(slides_metadata.values()),
                                     index=list(slides_metadata.keys()))

