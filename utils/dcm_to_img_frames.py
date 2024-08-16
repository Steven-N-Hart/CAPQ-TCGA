import os
import argparse
from minipath import Minipath, MagPairs, get_single_dcm_img, read_dicom
from PIL import Image
import io
from google.cloud import storage
import pandas as pd
import logging
from google.api_core.retry import Retry

logger = logging.getLogger()

class DicomProcessor:
    def __init__(self, bq_results_df, bucket_name, dirname):
        self.bq_results_df = bq_results_df
        self.bucket_name = bucket_name
        self.keys_to_keep = ['row_min', 'row_max', 'col_min', 'col_max', 'frame']
        self.dirname = dirname
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

    def get_or_create_bucket(self, bucket_name):
        """
        Get or create a Google Cloud Storage bucket.
        """
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
            logging.info(f"Bucket {bucket_name} already exists.")
        except storage.exceptions.NotFound:
            bucket = self.storage_client.create_bucket(bucket_name)
            logging.info(f"Bucket {bucket_name} created.")
        return bucket

    def process_dicom(self):
        for _, j in self.bq_results_df.iterrows():
            if j['row_num_asc'] != 1:
                continue
            gcs_url = j['gcs_url']
            dcm = read_dicom(gcs_url)
            grid_array = get_single_dcm_img(dcm)
            mp = Minipath(img=Image.fromarray(grid_array))
            results_dict = mp.rank_patches_for_diversity(explained_variance=0.95)
            logger.info(f"Minipath found {results_dict['n_clusters']} n_clusters")
            img_to_use_at_low_mag = [results_dict['patches_with_labels'][x] for x in results_dict['closest_samples_idx']]
            mag_pairs = MagPairs(dcm, img_to_use_at_low_mag=img_to_use_at_low_mag, bq_results_df=self.bq_results_df)
            clean_high_mag_frames = mag_pairs.clean_high_mag_frames
            self.upload_frames(clean_high_mag_frames, mag_pairs)

    def upload_frames(self, clean_high_mag_frames, mag_pairs):
        for i in range(len(clean_high_mag_frames)):
            fname_parts = [str(clean_high_mag_frames[i].get(x)) for x in self.keys_to_keep]
            fname = os.path.join(self.dirname, '_'.join([mag_pairs.high_mag_dcm.SOPInstanceUID, '_'.join(fname_parts)])) + '.png'
            img = Image.fromarray(clean_high_mag_frames[i]['img_array'])
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            blob = self.bucket.blob(fname)

            # Retry configuration
            retry_strategy = Retry(
                initial=1.0,  # initial delay in seconds
                maximum=30.0,  # maximum delay in seconds
                multiplier=2.0,  # exponential backoff multiplier
                deadline=300.0  # overall deadline in seconds
            )

            try:
                blob.upload_from_file(img_byte_arr, content_type='image/png', retry=retry_strategy)
                logger.info(f"Uploaded {fname} to GCS bucket {self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to upload {fname} to GCS bucket {self.bucket_name}: {e}")




def main(bq_results_csv, bucket_name, dirname):
    if isinstance(bq_results_csv, str):
        bq_results_df = pd.read_csv(bq_results_csv)
    elif isinstance(bq_results_csv, pd.DataFrame):
        bq_results_df = bq_results_csv
    else:
        raise TypeError('bq_results_csv should be a string or a pandas dataframe')
    processor = DicomProcessor(bq_results_df, bucket_name, dirname)
    processor.process_dicom()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process DICOM files and upload images to Google Cloud Storage.")
    parser.add_argument('--bq_results_csv', default='gs://capq-tcga/workspace/bq_results_df.csv', help='Path to the CSV file containing BQ results.')
    parser.add_argument('--bucket_name', default='capq-tcga', help='Name of the Google Cloud Storage bucket.')
    parser.add_argument('--dirname', default='images', help='Directory name for storing images.')
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    main(args.bq_results_csv, args.bucket_name, args.dirname)
