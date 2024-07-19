import pandas as pd
from typing import Any, Dict, List
from google.cloud import storage
import pydicom
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger()

def download_public_file(bucket_name, source_blob_name, destination_file_name):
    """Downloads a public blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    logging.info(
        "Downloaded public blob {} from bucket {} to {}.".format(
            source_blob_name, bucket.name, destination_file_name
        )
    )

def get_single_dcm_img(dcm_input) -> np.ndarray:
    if isinstance(dcm_input, str):
        dcm = pydicom.dcmread(dcm_input)
    elif isinstance(dcm_input, pydicom.dataset.FileDataset):
        dcm = dcm_input
    else:
        raise ValueError("Input must be a DICOM file path string or a pydicom DICOM object")
    # Extract necessary metadata
    total_pixel_matrix_columns = dcm.TotalPixelMatrixColumns
    total_pixel_matrix_rows = dcm.TotalPixelMatrixRows
    columns = dcm.Columns
    rows = dcm.Rows

    # Calculate grid size
    grid_cols = int(np.ceil(total_pixel_matrix_columns / columns))
    grid_rows = int(np.ceil(total_pixel_matrix_rows / rows))

    # Assuming your array is named 'frames' and has shape (~72, 256, 256, 3)
    frames = dcm.pixel_array

    # Create an empty array to hold the grid
    frame_height, frame_width, channels = frames.shape[1:]
    grid_array = np.zeros((grid_rows * frame_height, grid_cols * frame_width, channels), dtype=np.uint8)

    # Populate the grid array using nested loops
    frame_index = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if frame_index < frames.shape[0]:
                grid_array[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width,
                :] = \
                    frames[frame_index]
                frame_index += 1

    # Convert the array to an image
    return grid_array






