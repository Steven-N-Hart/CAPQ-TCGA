import pandas as pd
from google.cloud import bigquery
import re
import logging
import argparse
logger = logging.getLogger()


logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

def parse_args():
    parser = argparse.ArgumentParser(description='Process DICOM files and upload images to Google Cloud Storage.')
    parser.add_argument('--gcp_project_id', type=str, help='Google Cloud Project ID', default='correlation-aware-pq')
    parser.add_argument('--dataset_id', type=str, help='Google Cloud dataset ID', default='tcga')
    parser.add_argument('--new_table_id', type=str, help='Clinical table to create', default='clinical_data')
    parser.add_argument('--embedding_table_id', type=str, help='Name of embeddings table', default='phikon')

    parser.add_argument('--phenotype_file', type=str, help='Path to the CSV file containing BQ results.',
                        default='data/mmc2.xlsx')
    parser.add_argument('--bq_results_csv', type=str, help='Path to the CSV file containing BQ results.',
                        default='data/workspace_bq_results_df.csv')

    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()
    logger.setLevel(args.verbosity)
    return args


def read_file(f):
    if f.endswith('.csv'):
        file_object = pd.read_csv(f)
        elif f.endswith('.xlsx'):
        file_object = pd.read_excel(f)
    elif f.endswith('.xls'):
        file_object = pd.read_excel(f)
    else:
        raise ValueError('File type not supported')
    return file_object


# Function to sanitize column names
def sanitize_column_name(name):
    return re.sub(r'[^\w]', '_', name)


if __name__ == '__main__':
    args = parse_args()
    # Load your data into DataFrames
    mmc2 = read_file(args.phenotype_file)
    wsd = read_file(args.bq_results_csv)

    # Initialize the BigQuery client
    client = bigquery.Client()

    # Remove Normals from mmc2
    mmc2 = mmc2[mmc2['Tumor or Normal'] == 'Tumor']

    # Create a common identifier
    mmc2['PatientID'] = mmc2['CLID'].str.split('-').str[:3].str.join('-')

    # Perform the merge
    merged_df = pd.merge(wsd, mmc2, left_on='PatientID', right_on='PatientID', how='inner')

    # Sanitize the column names
    merged_df.columns = [sanitize_column_name(col) for col in merged_df.columns]

    # Convert the DataFrame to a BigQuery table
    job = client.load_table_from_dataframe(merged_df, f'{args.gcp_project_id}.{args.dataset_id}.{args.new_table_id}')

    # Wait for the job to complete
    job.result()
    logging.info(f"Loaded {job.output_rows} rows into {dataset_id}.{new_table_id}.")

    # Now perform the join directly in BigQuery and overwrite the clinical_data table
    query = f"""
        CREATE OR REPLACE TABLE `{args.gcp_project_id}.{args.dataset_id}.{args.new_table_id}` AS
        SELECT clinical_data.*, phikon.image_name, phikon.embedding
        FROM `{args.gcp_project_id}.{args.dataset_id}.{args.new_table_id}` AS clinical_data
        LEFT JOIN `{args.gcp_project_id}.{args.dataset_id}.{args.embedding_table_id}` AS phikon
        ON clinical_data.SOPInstanceUID = phikon.SOPInstanceUID
    """

    # Execute the query to overwrite the clinical_data table
    query_job = client.query(query)
    query_job.result()  # Wait for the query to finish

    logging.info(f"Table `{dataset_id}.{new_table_id}` has been updated with the merged data.")
